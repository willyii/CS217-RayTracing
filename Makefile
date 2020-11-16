.PHONY: build
build:
	mkdir -p build
	cd build && \
	cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .. && \
	make

.PHONY: debug
debug:
	mkdir -p build
	cd build && \
	cmake -DCMAKE_BUILD_TYPE=debug .. && \
	make

.PHONY: clean
clean:
	rm -rf build

