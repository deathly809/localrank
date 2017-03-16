
OUTPUT=../../../${OUTDIR}

DIRS=$(wildcard ./*/)
PROGS=$(patsubst ./%/,$(OUTPUT)/%,$(DIRS))

.PHONY: all


all: $(PROGS)

define build_rule
$(OUTPUT)/$1: $1/*.go
	go build -o $(OUTPUT)/$1 $1/*.go
endef

$(foreach f,$(patsubst ./%/,%,$(DIRS)),$(eval $(call build_rule,$f)))

clean:
	-@rm -fv $(PROGS)

nuke: clean
