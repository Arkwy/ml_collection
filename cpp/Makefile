# Compiler
HIPCC = hipcc

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Executable name
TARGET = $(BIN_DIR)/main

# Find all source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp $(SRC_DIR)/utils/*.cpp  $(SRC_DIR)/tensor/*.cpp)
PSO := $(wildcard $(SRC_DIR)/pso/*.cpp)
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Compiler flags
CXXFLAGS = -std=c++2b -O2 -Wall
LDFLAGS =

LOG_LEVEL := LOG_LEVEL_DEBUG

# Create necessary directories
$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR) $(OBJ_DIR)/utils $(OBJ_DIR)/tensor $(OBJ_DIR)/pso)

# Default target
build: $(TARGET)

# Build and run
brun: build
	./$(BIN_DIR)/main

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
