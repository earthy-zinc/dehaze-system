package import_export

import (
	"io"
)

type VirusScanner interface {
	Scan(reader io.Reader, fileName string) error
}

type NoOpVirusScanner struct{}

func (NoOpVirusScanner) Scan(io.Reader, string) error { return nil }
