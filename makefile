# Nome do executável
TARGET = somaVetores

# Compilador CUDA
NVCC = nvcc

# Flags de compilação
NVCCFLAGS = -arch=sm_52 -O2

# Regra padrão: compila e roda
all: $(TARGET)
	./$(TARGET)

# Como compilar
$(TARGET): $(TARGET).cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Limpar os arquivos gerados
clean:
	rm -f $(TARGET)
