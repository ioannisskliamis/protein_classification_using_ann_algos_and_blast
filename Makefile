# Compiler
CC = g++
CFLAGS = -Iinclude -O3 -march=native -ffast-math -std=c++17

# Directories
SRC_DIR = src
BUILD_DIR = build

# Executable
TARGET = search

# Common source files
SRCS = 	src/main.cpp \
		src/parsingFuncs.cpp \
		src/euclidean.cpp \
		src/Dataset.cpp \
		src/LSH.cpp \
		src/Hypercube.cpp \
		src/ivfbase.cpp \
		src/ivfflat.cpp \
		src/ivfpq.cpp \
		src/readFuncs.cpp

# Common object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

# Default target (all algorithms)
all: $(TARGET)

# Rule to create the 'search' executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Rule for compiling object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p build

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean