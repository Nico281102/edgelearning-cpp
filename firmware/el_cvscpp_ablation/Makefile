SHELL := /bin/sh

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(THIS_DIR)/../..)

PROJECT_ROOT ?=
EDGE_C_ROOT ?=
RLTOOLS_ROOT ?=
TOOLCHAIN_BIN ?=
APPLI_PROJECT_NAME ?= EL_C_vsCpp_Appli
BUILD_CONFIG ?= EL_C_vsCpp_Ablation_DEBUG

ifeq ($(strip $(PROJECT_ROOT)),)
$(error PROJECT_ROOT is required; set it in firmware/el_cvscpp_ablation/.env or pass PROJECT_ROOT=/path/to/EL_C_vsCpp)
endif

ifeq ($(strip $(TOOLCHAIN_BIN)),)
$(error TOOLCHAIN_BIN is required; set it in firmware/el_cvscpp_ablation/.env or pass TOOLCHAIN_BIN=/path/to/arm-none-eabi/bin)
endif

H1 ?= 8
H2 ?= 8
INPUT_FEATURES ?= 3
CONFIG_SUFFIX := $(INPUT_FEATURES)_$(H1)x$(H2)_1
VARIANT ?= cpp_m55
SUPPORTED_VARIANTS := all legacy_c cpp_direct_c_backend cpp_m55 cpp_generic rltools_generic
LEGACY_C_REQUIRED_VARIANTS := all legacy_c cpp_direct_c_backend

ifeq ($(filter $(VARIANT),$(SUPPORTED_VARIANTS)),)
$(error VARIANT must be one of: $(SUPPORTED_VARIANTS))
endif

ifneq ($(filter $(VARIANT),$(LEGACY_C_REQUIRED_VARIANTS)),)
ifeq ($(strip $(EDGE_C_ROOT)),)
$(error EDGE_C_ROOT is required for VARIANT=$(VARIANT); set EL_CVSCPP_EDGE_C_ROOT in firmware/el_cvscpp_ablation/.env or pass EDGE_C_ROOT=/path/to/legacy/EdgeLearning)
endif
endif

VARIANT_ID := 0
ifeq ($(VARIANT),legacy_c)
VARIANT_ID := 1
endif
ifeq ($(VARIANT),cpp_direct_c_backend)
VARIANT_ID := 2
endif
ifeq ($(VARIANT),cpp_m55)
VARIANT_ID := 3
endif
ifeq ($(VARIANT),cpp_generic)
VARIANT_ID := 4
endif
ifeq ($(VARIANT),rltools_generic)
VARIANT_ID := 5
endif

RLTOOLS_REQUIRED_VARIANTS := all rltools_generic
ifneq ($(filter $(VARIANT),$(RLTOOLS_REQUIRED_VARIANTS)),)
ifeq ($(strip $(RLTOOLS_ROOT)),)
$(error RLTOOLS_ROOT is required for VARIANT=$(VARIANT); set EL_CVSCPP_RLTOOLS_ROOT in firmware/el_cvscpp_ablation/.env or pass RLTOOLS_ROOT=/path/to/rl-tools)
endif
endif

APPLI_DIR := $(PROJECT_ROOT)/STM32CubeIDE/Appli
BUILD_ROOT ?= $(APPLI_DIR)/$(BUILD_CONFIG)
ifeq ($(VARIANT),all)
BUILD_DIR := $(BUILD_ROOT)/$(CONFIG_SUFFIX)
TARGET := $(APPLI_PROJECT_NAME)_$(CONFIG_SUFFIX)
else
BUILD_DIR := $(BUILD_ROOT)/$(CONFIG_SUFFIX)/$(VARIANT)
TARGET := $(APPLI_PROJECT_NAME)_$(CONFIG_SUFFIX)_$(VARIANT)
endif
OBJ_DIR := $(BUILD_DIR)/obj

ELF := $(BUILD_DIR)/$(TARGET).elf
MAP := $(BUILD_DIR)/$(TARGET).map
BIN := $(BUILD_DIR)/$(TARGET).bin
LIST := $(BUILD_DIR)/$(TARGET).list
SIZE_TXT := $(BUILD_DIR)/$(TARGET).size.txt

CC := $(TOOLCHAIN_BIN)/arm-none-eabi-gcc
CXX := $(TOOLCHAIN_BIN)/arm-none-eabi-g++
OBJCOPY := $(TOOLCHAIN_BIN)/arm-none-eabi-objcopy
OBJDUMP := $(TOOLCHAIN_BIN)/arm-none-eabi-objdump
SIZE := $(TOOLCHAIN_BIN)/arm-none-eabi-size

LINKER_SCRIPT := $(APPLI_DIR)/STM32N657X0HXQ_ROMxspi2.ld

COMMON_DEFS := \
  -DAPP_EL_C_VSCPP_ABLATION \
  -DEL_CVSCPP_INPUT_FEATURES=$(INPUT_FEATURES) \
  -DEL_CVSCPP_H1=$(H1) \
  -DEL_CVSCPP_H2=$(H2) \
  -DEL_CVSCPP_VARIANT=$(VARIANT_ID) \
  -DDEBUG \
  -DUSE_HAL_DRIVER \
  -DSTM32N657xx \
  -DARM_MATH_MVEF \
  -DARM_MATH_DSP \
  -DARM_MATH_LOOPUNROLL \
  -DUSE_FULL_LL_DRIVER

COMMON_INCS := \
  -I$(REPO_ROOT)/firmware/el_cvscpp_ablation \
  -I$(REPO_ROOT)/include \
  -I$(PROJECT_ROOT)/Appli/Core/Inc \
  -I$(PROJECT_ROOT)/Secure_nsclib \
  -I$(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Inc \
  -I$(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Inc/Legacy \
  -I$(PROJECT_ROOT)/Drivers/CMSIS/Device/ST/STM32N6xx/Include \
  -I$(PROJECT_ROOT)/Drivers/CMSIS/Include \
  -I$(PROJECT_ROOT)/Drivers/DSP/Include \
  $(if $(strip $(EDGE_C_ROOT)),-I$(EDGE_C_ROOT) -I$(EDGE_C_ROOT)/Inc,) \
  $(if $(strip $(RLTOOLS_ROOT)),-I$(RLTOOLS_ROOT)/include,)

CPU_FLAGS := -mcpu=cortex-m55 -mfpu=auto -mfloat-abi=hard -mthumb -mcmse
OPT_FLAGS := -Ofast -ffunction-sections -fdata-sections -flax-vector-conversions
WARN_FLAGS := -Wall -Wextra -Wno-missing-field-initializers
CXX_WARN_FLAGS := $(WARN_FLAGS) -Wno-unused-parameter -Wno-unused-local-typedefs
DEP_FLAGS := -MMD -MP

CFLAGS := $(CPU_FLAGS) -std=gnu11 $(OPT_FLAGS) $(WARN_FLAGS) $(COMMON_DEFS) $(COMMON_INCS) --specs=nano.specs $(DEP_FLAGS)
CXXFLAGS := $(CPU_FLAGS) -std=gnu++20 $(OPT_FLAGS) $(CXX_WARN_FLAGS) -fno-exceptions -fno-rtti -fno-use-cxa-atexit $(COMMON_DEFS) $(COMMON_INCS) --specs=nano.specs $(DEP_FLAGS)
ASFLAGS := $(CPU_FLAGS) -x assembler-with-cpp $(COMMON_DEFS) --specs=nano.specs $(DEP_FLAGS)
LDFLAGS := $(CPU_FLAGS) -T$(LINKER_SCRIPT) --specs=nosys.specs --specs=nano.specs -Wl,-Map=$(MAP) -Wl,--gc-sections -static -Wl,--cmse-implib -Wl,--out-implib=$(BUILD_DIR)/secure_nsclib.o
LDLIBS := -Wl,--start-group -lc -lm -lstdc++ -lsupc++ -Wl,--end-group

C_SRCS := \
  $(PROJECT_ROOT)/Appli/Core/Src/main.c \
  $(PROJECT_ROOT)/Appli/Core/Src/system_stm32n6xx_s.c \
  $(PROJECT_ROOT)/Appli/Core/Src/stm32n6xx_hal_msp.c \
  $(PROJECT_ROOT)/Appli/Core/Src/stm32n6xx_it.c \
  $(PROJECT_ROOT)/Appli/Core/Src/secure_nsc.c \
  $(APPLI_DIR)/Application/User/Core/syscalls.c \
  $(APPLI_DIR)/Application/User/Core/sysmem.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_cacheaxi.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_cortex.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_dma_ex.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_exti.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_gpio.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_pwr_ex.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rcc_ex.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_rif.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_uart.c \
  $(PROJECT_ROOT)/Drivers/STM32N6xx_HAL_Driver/Src/stm32n6xx_hal_uart_ex.c \
  $(PROJECT_ROOT)/Drivers/DSP/Source/BasicMathFunctions/arm_add_f32.c \
  $(PROJECT_ROOT)/Drivers/DSP/Source/MatrixFunctions/arm_mat_init_f32.c \
  $(PROJECT_ROOT)/Drivers/DSP/Source/MatrixFunctions/arm_mat_vec_mult_f32.c \
  $(PROJECT_ROOT)/Drivers/DSP/Source/SupportFunctions/arm_fill_f32.c

LEGACY_C_SRCS := \
  $(EDGE_C_ROOT)/0_Common/el_config.c \
  $(EDGE_C_ROOT)/0_Common/el_platform_stm32n657xx.c \
  $(EDGE_C_ROOT)/0_Common/el_system.c \
  $(EDGE_C_ROOT)/1_Engine/el_backend_m55.c \
  $(EDGE_C_ROOT)/1_Engine/el_model.c \
  $(EDGE_C_ROOT)/2_Network/el_layer.c \
  $(EDGE_C_ROOT)/2_Network/el_layer_factory.c \
  $(EDGE_C_ROOT)/2_Network/el_network.c \
  $(EDGE_C_ROOT)/2_Network/el_network_factory.c \
  $(EDGE_C_ROOT)/2_Network/el_network_params.c

ifneq ($(filter $(VARIANT),$(LEGACY_C_REQUIRED_VARIANTS)),)
C_SRCS += $(LEGACY_C_SRCS)
endif

CPP_SRCS := \
  $(REPO_ROOT)/firmware/el_cvscpp_ablation/el_cvscpp_ablation_run.cpp

S_SRCS := \
  $(APPLI_DIR)/Application/User/Startup/startup_stm32n657x0hxq.s

C_OBJS := $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(basename $(notdir $(C_SRCS)))))
CPP_OBJS := $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(basename $(notdir $(CPP_SRCS)))))
S_OBJS := $(addprefix $(OBJ_DIR)/,$(addsuffix .o,$(basename $(notdir $(S_SRCS)))))
OBJS := $(C_OBJS) $(CPP_OBJS) $(S_OBJS)
DEPS := $(OBJS:.o=.d)

.PHONY: all clean size print-config

all: $(ELF) $(BIN) $(LIST) $(SIZE_TXT)

print-config:
	@printf 'PROJECT_ROOT=%s\n' '$(PROJECT_ROOT)'
	@printf 'EDGE_C_ROOT=%s\n' '$(EDGE_C_ROOT)'
	@printf 'RLTOOLS_ROOT=%s\n' '$(RLTOOLS_ROOT)'
	@printf 'TOOLCHAIN_BIN=%s\n' '$(TOOLCHAIN_BIN)'
	@printf 'APPLI_PROJECT_NAME=%s\n' '$(APPLI_PROJECT_NAME)'
	@printf 'BUILD_CONFIG=%s\n' '$(BUILD_CONFIG)'
	@printf 'VARIANT=%s\n' '$(VARIANT)'
	@printf 'BUILD_DIR=%s\n' '$(BUILD_DIR)'
	@printf 'ELF=%s\n' '$(ELF)'
	@printf 'INPUT_FEATURES=%s H1=%s H2=%s\n' '$(INPUT_FEATURES)' '$(H1)' '$(H2)'

$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

define C_RULE
$(OBJ_DIR)/$(basename $(notdir $(1))).o: $(1) | $(OBJ_DIR)
	$$(CC) "$$<" $$(CFLAGS) -MF"$$(@:.o=.d)" -MT"$$@" -c -o "$$@"
endef

define CPP_RULE
$(OBJ_DIR)/$(basename $(notdir $(1))).o: $(1) | $(OBJ_DIR)
	$$(CXX) "$$<" $$(CXXFLAGS) -MF"$$(@:.o=.d)" -MT"$$@" -c -o "$$@"
endef

define S_RULE
$(OBJ_DIR)/$(basename $(notdir $(1))).o: $(1) | $(OBJ_DIR)
	$$(CC) "$$<" $$(ASFLAGS) -MF"$$(@:.o=.d)" -MT"$$@" -c -o "$$@"
endef

$(foreach src,$(C_SRCS),$(eval $(call C_RULE,$(src))))
$(foreach src,$(CPP_SRCS),$(eval $(call CPP_RULE,$(src))))
$(foreach src,$(S_SRCS),$(eval $(call S_RULE,$(src))))

$(ELF): $(OBJS) $(LINKER_SCRIPT) | $(BUILD_DIR)
	$(CXX) -o "$@" $(OBJS) $(LDFLAGS) $(LDLIBS)

$(BIN): $(ELF)
	$(OBJCOPY) -O binary "$<" "$@"

$(LIST): $(ELF)
	$(OBJDUMP) -h -S "$<" > "$@"

$(SIZE_TXT): $(ELF)
	$(SIZE) "$<" | tee "$@"

size: $(SIZE_TXT)

clean:
	rm -rf "$(BUILD_ROOT)"

-include $(DEPS)
