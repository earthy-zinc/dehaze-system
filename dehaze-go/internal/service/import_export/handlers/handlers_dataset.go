package handlers

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/earthyzinc/dehaze-go/internal/model"
	"github.com/earthyzinc/dehaze-go/internal/service/import_export"
	"gorm.io/gorm"
)

const (
	datasetStructureByItem   = "by_item"
	datasetDefaultFileExt    = ".jpg"
	datasetZipBufferSize     = 8192
	datasetThumbnailSubfolder = "thumbnail"
)

type DatasetExportHandler struct {
	import_export.BaseExportHandler
	db      *gorm.DB
	storage StorageDownloader
}

type StorageDownloader interface {
	Download(ctx context.Context, objectName string) (io.ReadCloser, error)
}

func NewDatasetExportHandler(db *gorm.DB, storage StorageDownloader) *DatasetExportHandler {
	return &DatasetExportHandler{db: db, storage: storage}
}

func (h *DatasetExportHandler) GetModule() string { return "dataset" }

func (h *DatasetExportHandler) UseDirectExport() bool { return true }

func (h *DatasetExportHandler) EstimateCount(params map[string]interface{}) int64 {
	if len(params) == 0 {
		return 0
	}
	items := h.resolveItems(params)
	if len(items) == 0 {
		return 0
	}
	itemIDs := make([]int64, 0, len(items))
	for _, it := range items {
		itemIDs = append(itemIDs, it.ID)
	}
	var count int64
	h.db.Model(&model.SysItemFile{}).Where("item_id IN ?", itemIDs).Count(&count)
	return count
}

func (h *DatasetExportHandler) GetFieldConfigs() []import_export.ExportFieldConfig {
	return []import_export.ExportFieldConfig{
		{Field: "datasetName", Label: "数据集名称", Order: 1},
		{Field: "itemName", Label: "数据项名称", Order: 2},
		{Field: "fileType", Label: "文件类型", Order: 3},
		{Field: "fileName", Label: "文件名", Order: 4},
		{Field: "fileSize", Label: "文件大小", Order: 5},
	}
}

func (h *DatasetExportHandler) GetDataProvider(ctx *import_export.ExportContext) import_export.ExportDataProvider {
	return &emptyDataProvider{}
}

type emptyDataProvider struct{}

func (emptyDataProvider) FetchBatch(int, int) [][]interface{} { return nil }

type datasetExportOptions struct {
	structure       string
	includeTypes    []string
	includeThumbnail bool
}

func parseDatasetOptions(params map[string]interface{}) datasetExportOptions {
	opts := datasetExportOptions{
		structure:       datasetStructureByItem,
		includeThumbnail: false,
	}
	if optsRaw, ok := params["options"].(map[string]interface{}); ok {
		if s, ok := optsRaw["structure"].(string); ok && s != "" {
			opts.structure = s
		}
		if list, ok := optsRaw["includeTypes"].([]interface{}); ok {
			for _, v := range list {
				if s, ok := v.(string); ok {
					opts.includeTypes = append(opts.includeTypes, s)
				}
			}
		}
		if b, ok := optsRaw["includeThumbnail"].(bool); ok {
			opts.includeThumbnail = b
		}
	}
	return opts
}

func (h *DatasetExportHandler) Export(ctx *import_export.ExportContext, callback import_export.ProgressCallback) error {
	zipWriter := zip.NewWriter(ctx.OutputStream)
	defer zipWriter.Close()

	params := ctx.QueryParams
	if len(params) == 0 {
		return nil
	}

	options := parseDatasetOptions(params)
	items := h.resolveItems(params)
	if len(items) == 0 {
		return nil
	}

	itemIDs := make([]int64, 0, len(items))
	for _, it := range items {
		itemIDs = append(itemIDs, it.ID)
	}

	var itemFiles []model.SysItemFile
	h.db.Where("item_id IN ?", itemIDs).Find(&itemFiles)

	itemFilesMap := make(map[int64][]model.SysItemFile)
	for _, f := range itemFiles {
		itemFilesMap[f.ItemID] = append(itemFilesMap[f.ItemID], f)
	}

	totalFiles := 0
	for _, item := range items {
		fileCount := len(itemFilesMap[item.ID])
		totalFiles += fileCount
		if options.includeThumbnail {
			totalFiles += fileCount
		}
	}

	callback.UpdateProgress(0, totalFiles, "开始导出数据集文件")

	fileIDs := make([]int64, 0, len(itemFiles))
	for _, f := range itemFiles {
		fileIDs = append(fileIDs, f.FileID)
	}
	fileMap := make(map[int64]model.SysFile)
	if len(fileIDs) > 0 {
		var files []model.SysFile
		h.db.Where("id IN ?", fileIDs).Find(&files)
		for _, f := range files {
			fileMap[int64(f.ID)] = f
		}
	}

	processedFiles := 0

	for _, item := range items {
		if callback.IsCancelled() {
			break
		}

		files := itemFilesMap[item.ID]
		for _, itemFile := range files {
			if callback.IsCancelled() {
				break
			}

			if shouldIncludeType(options.includeTypes, itemFile.Type) {
				if err := h.addFileToZip(ctx, zipWriter, itemFile, options.structure, item.Name, ""); err != nil {
					return err
				}
				processedFiles++
				callback.UpdateProgress(processedFiles, totalFiles, "正在导出: "+item.Name)
			}

			if options.includeThumbnail && itemFile.ThumbnailFileID != nil {
				thumbFile, ok := fileMap[*itemFile.ThumbnailFileID]
				if !ok || thumbFile.ObjectName == "" {
					continue
				}
				if err := h.addThumbnailToZip(ctx, zipWriter, thumbFile, options.structure, item.Name); err != nil {
					return err
				}
				processedFiles++
				callback.UpdateProgress(processedFiles, totalFiles, "导出缩略图")
			}
		}
	}
	return nil
}

func (h *DatasetExportHandler) resolveItems(params map[string]interface{}) []model.SysDatasetItem {
	if itemIDsRaw, ok := params["itemIds"].([]interface{}); ok && len(itemIDsRaw) > 0 {
		itemIDs := make([]int64, 0, len(itemIDsRaw))
		for _, v := range itemIDsRaw {
			switch n := v.(type) {
			case float64:
				itemIDs = append(itemIDs, int64(n))
			case int:
				itemIDs = append(itemIDs, int64(n))
			case int64:
				itemIDs = append(itemIDs, n)
			}
		}
		if len(itemIDs) == 0 {
			return nil
		}
		var items []model.SysDatasetItem
		h.db.Where("id IN ?", itemIDs).Find(&items)
		return items
	}

	if v, ok := params["itemId"]; ok {
		itemID := toInt64(v)
		if itemID > 0 {
			var item model.SysDatasetItem
			if err := h.db.Where("id = ?", itemID).First(&item).Error; err == nil {
				return []model.SysDatasetItem{item}
			}
			return nil
		}
	}

	datasetIDRaw, ok := params["datasetId"]
	if !ok {
		datasetIDRaw = params["targetId"]
	}
	if datasetIDRaw != nil {
		datasetID := toInt64(datasetIDRaw)
		if datasetID > 0 {
			tx := h.db.Model(&model.SysDatasetItem{}).Where("dataset_id = ?", datasetID)
			if filters, ok := params["filters"].(map[string]interface{}); ok {
				if name, ok := filters["name"].(string); ok && name != "" {
					tx = tx.Where("name LIKE ?", "%"+name+"%")
				}
			}
			var items []model.SysDatasetItem
			tx.Find(&items)
			return items
		}
	}

	return nil
}

func toInt64(v interface{}) int64 {
	switch n := v.(type) {
	case int:
		return int64(n)
	case int8:
		return int64(n)
	case int16:
		return int64(n)
	case int32:
		return int64(n)
	case int64:
		return n
	case float32:
		return int64(n)
	case float64:
		return int64(n)
	}
	return 0
}

func shouldIncludeType(includeTypes []string, fileType string) bool {
	if len(includeTypes) == 0 {
		return true
	}
	for _, t := range includeTypes {
		if t == fileType {
			return true
		}
	}
	return false
}

func (h *DatasetExportHandler) addFileToZip(ctx *import_export.ExportContext, zipWriter *zip.Writer, itemFile model.SysItemFile, structure, itemName, subfolder string) error {
	var sysFile model.SysFile
	if err := h.db.Where("id = ?", itemFile.FileID).First(&sysFile).Error; err != nil {
		return nil
	}
	if sysFile.ObjectName == "" {
		return nil
	}

	entryPath := buildZipEntryPath(structure, itemName, subfolder, itemFile.ID, sysFile.Name)
	zipEntry := &zip.FileHeader{Name: entryPath, Method: zip.Deflate}
	writer, err := zipWriter.CreateHeader(zipEntry)
	if err != nil {
		return err
	}

	reader, err := h.storage.Download(ctx.Ctx, sysFile.ObjectName)
	if err != nil {
		return err
	}
	defer reader.Close()

	buf := make([]byte, datasetZipBufferSize)
	for {
		n, err := reader.Read(buf)
		if n > 0 {
			if _, werr := writer.Write(buf[:n]); werr != nil {
				return werr
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (h *DatasetExportHandler) addThumbnailToZip(ctx *import_export.ExportContext, zipWriter *zip.Writer, sysFile model.SysFile, structure, itemName string) error {
	if sysFile.ObjectName == "" {
		return nil
	}
	entryPath := buildZipEntryPath(structure, itemName, datasetThumbnailSubfolder, int64(sysFile.ID), sysFile.Name)
	zipEntry := &zip.FileHeader{Name: entryPath, Method: zip.Deflate}
	writer, err := zipWriter.CreateHeader(zipEntry)
	if err != nil {
		return err
	}

	reader, err := h.storage.Download(ctx.Ctx, sysFile.ObjectName)
	if err != nil {
		return err
	}
	defer reader.Close()

	buf := make([]byte, datasetZipBufferSize)
	for {
		n, err := reader.Read(buf)
		if n > 0 {
			if _, werr := writer.Write(buf[:n]); werr != nil {
				return werr
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func buildZipEntryPath(structure, itemName, subfolder string, fileID int64, fileName string) string {
	extension := getFileExtension(fileName)
	baseName := fmt.Sprintf("%d%s", fileID, extension)
	if structure == datasetStructureByItem {
		if subfolder != "" {
			return strings.TrimSpace(itemName+"/"+subfolder+"/"+baseName)
		}
		return strings.TrimSpace(itemName + "/" + baseName)
	}
	if subfolder != "" {
		return strings.TrimSpace(subfolder + "/" + baseName)
	}
	return strings.TrimSpace(baseName)
}

func getFileExtension(fileName string) string {
	if fileName == "" {
		return datasetDefaultFileExt
	}
	dotIdx := strings.LastIndex(fileName, ".")
	if dotIdx > 0 {
		return fileName[dotIdx:]
	}
	return datasetDefaultFileExt
}
