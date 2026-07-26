package import_export

import "fmt"

type ExportHandlerRegistry struct {
	handlers map[string]ExportHandler
}

func NewExportHandlerRegistry(handlers []ExportHandler) *ExportHandlerRegistry {
	r := &ExportHandlerRegistry{handlers: make(map[string]ExportHandler, len(handlers))}
	for _, h := range handlers {
		if existing, ok := r.handlers[h.GetModule()]; ok {
			panic(fmt.Sprintf("duplicate ExportHandler for module %s: %T vs %T", h.GetModule(), existing, h))
		}
		r.handlers[h.GetModule()] = h
	}
	return r
}

func (r *ExportHandlerRegistry) GetHandler(module string) (ExportHandler, error) {
	h, ok := r.handlers[module]
	if !ok {
		return nil, NewModuleNotSupportedError("export", module)
	}
	return h, nil
}

type ImportHandlerRegistry struct {
	handlers map[string]ImportHandler
}

func NewImportHandlerRegistry(handlers []ImportHandler) *ImportHandlerRegistry {
	r := &ImportHandlerRegistry{handlers: make(map[string]ImportHandler, len(handlers))}
	for _, h := range handlers {
		if existing, ok := r.handlers[h.GetModule()]; ok {
			panic(fmt.Sprintf("duplicate ImportHandler for module %s: %T vs %T", h.GetModule(), existing, h))
		}
		r.handlers[h.GetModule()] = h
	}
	return r
}

func (r *ImportHandlerRegistry) GetHandler(module string) (ImportHandler, error) {
	h, ok := r.handlers[module]
	if !ok {
		return nil, NewModuleNotSupportedError("import", module)
	}
	return h, nil
}
