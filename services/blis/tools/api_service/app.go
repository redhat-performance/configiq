package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sort"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

const defaultHWConfigURL = "https://raw.githubusercontent.com/inference-sim/inference-sim/main/hardware_config.json"

var (
	hwCatalog     map[string]sim.HardwareCalib
	hwCatalogOnce sync.Once
	hwCatalogErr  error
)

// ensureHWCatalog loads the hardware catalog exactly once on first call.
// Subsequent calls return the cached result. Thread-safe.
func ensureHWCatalog() (map[string]sim.HardwareCalib, error) {
	hwCatalogOnce.Do(func() {
		data, err := loadHardwareConfig()
		if err != nil {
			hwCatalogErr = err
			return
		}

		parsed, err := latency.ParseHardwareCalibEntries(data)
		if err != nil {
			hwCatalogErr = fmt.Errorf("parse hardware config: %w", err)
			return
		}
		if len(parsed) == 0 {
			hwCatalogErr = fmt.Errorf("hardware config is empty")
			return
		}

		hwCatalog = parsed
		log.Printf("loaded %d GPU systems from hardware config", len(hwCatalog))
	})
	return hwCatalog, hwCatalogErr
}

func loadHardwareConfig() ([]byte, error) {
	data, err := os.ReadFile("hardware_config.json")
	if err == nil {
		log.Println("using local hardware_config.json")
		return data, nil
	}

	if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read local hardware_config.json: %w", err)
	}

	log.Printf("local hardware_config.json not found, fetching from %s", defaultHWConfigURL)
	return fetchFromURL(defaultHWConfigURL)
}

func fetchFromURL(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetch %s: HTTP %d", url, resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response from %s: %w", url, err)
	}
	return data, nil
}

// SystemResponse matches the aisimulators /systems?include=specs response shape.
type SystemResponse struct {
	ID                   string  `json:"id"`
	Name                 string  `json:"name"`
	MemoryBytes          int64   `json:"memory_bytes"`
	MemoryBandwidthBytes int64   `json:"memory_bandwidth_bytes"`
	BF16TFlops           float64 `json:"bf16_tflops"`
	FP8TFlops            float64 `json:"fp8_tflops"`
	MFUPrefill           float64 `json:"mfu_prefill"`
	MFUDecode            float64 `json:"mfu_decode"`
	IntraNodeBwGBps      float64 `json:"intra_node_bw_gbps"`
	InterNodeBwGBps      float64 `json:"inter_node_bw_gbps"`
}

func getSystems(c *gin.Context) {
	catalog, err := ensureHWCatalog()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	systems := make([]SystemResponse, 0, len(catalog))

	for name, hw := range catalog {
		systems = append(systems, SystemResponse{
			ID:                   name,
			Name:                 name,
			MemoryBytes:          int64(hw.MemoryGiB * 1024 * 1024 * 1024),
			MemoryBandwidthBytes: int64(hw.BwPeakTBs * 1e12),
			BF16TFlops:           hw.TFlopsPeak,
			FP8TFlops:            hw.TFlopsFP8,
			MFUPrefill:           hw.MfuPrefill,
			MFUDecode:            hw.MfuDecode,
			IntraNodeBwGBps:      hw.IntraNodeBwGBps,
			InterNodeBwGBps:      hw.InterNodeBwGBps,
		})
	}

	sort.Slice(systems, func(i, j int) bool {
		return systems[i].ID < systems[j].ID
	})

	c.JSON(http.StatusOK, gin.H{"systems": systems})
}

func main() {
	router := gin.Default()
	router.GET("/backends", getBackends)
	router.GET("/models", getModels)
	router.GET("/systems", getSystems)
	router.GET("/metrics", getMetrics)
	router.POST("/memory", postMemory)
	router.POST("/recommend", postRecommend)
	router.POST("/predict", postPredict)
	router.POST("/estimate", postPredict)
	router.Run(":8080")
}

func getBackends(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"backends": []string{"llama", "gpt", "gpt-4o"}})
}

func getModels(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"models": []string{"llama", "gpt", "gpt-4o"}})
}

func getMetrics(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"metrics": []string{"llama", "gpt", "gpt-4o"}})
}

func postMemory(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"memory": []string{"llama", "gpt", "gpt-4o"}})
}

func postRecommend(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"recommend": []string{"llama", "gpt", "gpt-4o"}})
}

func postPredict(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"predict": []string{"llama", "gpt", "gpt-4o"}})
}
