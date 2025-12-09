A simple e2e example of MDR.
============================

## 0. Env setup

```bash
source ../src.env
```

## 1. Build the client application and run it.

```bash
make
./vec_add_module
```

## 2. Run MDR to raise GPU kernel from ISA to MLIR then back to ISA. Add printf() logic into the kernel.

```bash
../mdr.py
```

## 3. Assemble modified GPU kernel.

```bash
make assemble
```

## 4. Run the application again.

```bash
./vec_add_module
```
