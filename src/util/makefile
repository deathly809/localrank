
LIB=../
INCL	:=-I$(LIB)
OUTPUT=../../${OUTDIR}

FLAGS	+=-Xptxas="-v"

CPP_OBJS := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
CU_OBJS  := $(patsubst %.cu,%.o,$(wildcard *.cu))
OBJECTS  := $(addprefix $(OUTPUT)/,${CPP_OBJS} ${CU_OBJS})
.PHONY: all

all:  $(OBJECTS)

$(OUTPUT)/%.o : %.cpp %.h
	nvcc -dc $(ARCH) $(FLAGS) $(STD) $(INCL) $< -odir $(OUTPUT)

$(OUTPUT)/%.o : %.cu %.cuh
	nvcc -dc $(ARCH) $(FLAGS) $(STD) $(INCL) $< -odir $(OUTPUT)

$(OUTPUT):
	mkdir $(OUTPUT)

clean:
	-@rm -fv $(OBJECTS)

nuke: clean
