# Compiler
HIPCC = hipcc

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Executable name
TARGET = $(BIN_DIR)/main

# Find all source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/utils/*.cpp  $(SRC_DIR)/array/*.cpp $(SRC_DIR)/ops/*.cpp $(SRC_DIR)/ops/reduce/*.cpp)
PSO := $(wildcard $(SRC_DIR)/pso/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Compiler flags
CXXFLAGS = -std=c++2b -O2 -Wall
LDFLAGS = -lrocrand 

LOG_LEVEL := LOG_LEVEL_DEBUG

# Create necessary directories
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR) $(OBJ_DIR)/utils $(OBJ_DIR)/array $(OBJ_DIR)/ops $(OBJ_DIR)/ops/reduce $(OBJ_DIR)/pso)

# Run (and build if needed) (default)
run: build
	./$(BIN_DIR)/main

# Build
build: $(TARGET)

# Build executable
$(TARGET): $(OBJS)
	$(HIPCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(HIPCC) $(CXXFLAGS) -DLOG_LEVEL=$(LOG_LEVEL) -c -o $@ $<

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: build run clean
