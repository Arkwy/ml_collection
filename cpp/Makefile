HIPCC = hipcc
TARGET = hip_build

all: $(TARGET)

$(TARGET): src/main.cpp
	$(HIPCC) -o $(TARGET) src/main.cpp

clean:
	rm -f $(TARGET)
