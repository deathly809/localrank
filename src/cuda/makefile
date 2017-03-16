SRC		:=../
LIB		:=$(SRC)
INCL	:=-I$(LIB)
OUTPUT	:=../../${OUTDIR}

FLAGS 	+=-dc -Xptxas="-v"

$(OUTPUT)/cudbscan.o: cudbscan.cu cudbscan.cuh $(SRC)/util/cudaUtil.cuh
	nvcc $(FLAGS) $(ARCH) $(STD) $(INCL) cudbscan.cu -odir $(OUTPUT)

clean:
	-@rm -fv $(OUTPUT)/cudbscan.o

nuke: clean
