
export OUTDIR :=./output

# Executable names
export EXE  	:=runner

# Global flags
SRC		:=./src
LIB		:=$(SRC)

ifeq ($(BucketSize),)
	BucketSize := 2
endif

ifeq ($(Big),"true")
	BIG := -DBIG_BENCH
else
	BIG := 
endif

ifeq ($(Experimental),"true")
	echo "Experimental enabled"
	EXP := -DEXPERIMENTAL
else
	EXP :=
endif

ifeq ($(Debug),"true")
	OPT := -g -G -pg
else
	OPT := -O3
endif

ifeq ($(Verify),"true")
	VERIFY := -DVERIFY
else
	VERIFY :=
endif

# Compiler flags
export ARCH		:=-arch=sm_35
export STD		:=--std=c++11
export FLAGS	:=-D_FORCE_INLINES $(VERIFY) $(OPT) $(BIG) $(EXP) -D BUCKET_SIZE=$(BucketSize)

INCL	:=-I$(LIB)

# Make targets work as intended
.PHONY: all
.PHONY: clean
.PHONY: test
.PHONY: exe
.PHONY: dbscan
.PHONY: util

# Makes all libraries and executables
all: $(OUTDIR) util dbscan exe test   

test:
	cd $(SRC)/tests/ && ${MAKE}

test_debug:
	cd $(SRC)/tests/ && ${MAKE} debug

test_big:
	cd $(SRC)/tests/ && ${MAKE} big

exe:
	cd $(SRC)/bin/ && ${MAKE}
	cd $(SRC)/go/cmd/ && ${MAKE}

dbscan:
	cd $(SRC)/cuda/ && ${MAKE}

util:
	cd $(SRC)/util/ && ${MAKE}

# Make sure the output directory exists
$(OUTDIR):
	mkdir $(OUTDIR)

nuke:
	make -C ./src nuke
