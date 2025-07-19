CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra -std=c99
LDFLAGS = -static -lopenblas -lm -flto

# Source files
SOURCES = train.c ssm.c data.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = train.out

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

# Build object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	@time ./$(TARGET)

# Clean build artifacts
clean:
	rm -f *.o *.out *.csv *.bin

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all     - Build the main executable (default)"
	@echo "  run     - Build and run the program"
	@echo "  clean   - Remove build artifacts"
	@echo "  debug   - Build with debug symbols"
	@echo "  help    - Show this help message"

.PHONY: all run clean debug help