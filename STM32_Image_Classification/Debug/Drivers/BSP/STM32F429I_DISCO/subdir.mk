################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery.c \
../Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_io.c \
../Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_lcd.c \
../Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_sdram.c \
../Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_ts.c 

OBJS += \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery.o \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_io.o \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_lcd.o \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_sdram.o \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_ts.o 

C_DEPS += \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery.d \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_io.d \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_lcd.d \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_sdram.d \
./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_ts.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/BSP/STM32F429I_DISCO/%.o: ../Drivers/BSP/STM32F429I_DISCO/%.c Drivers/BSP/STM32F429I_DISCO/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DUSE_HAL_DRIVER -DDEBUG -DSTM32F429xx -c -I../Core/Inc -I../USB_HOST/App -I../USB_HOST/Target -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Middlewares/ST/STM32_USB_Host_Library/Core/Inc -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../FATFS/Target -I../FATFS/App -I../LIBJPEG/App -I../LIBJPEG/Target -I../X-CUBE-AI/App -I../X-CUBE-AI -I../Middlewares/ST/AI/Inc -I../Middlewares/Third_Party/FatFs/src -I../Middlewares/Third_Party/LibJPEG/include -I../Middlewares/ST/STM32_USB_Host_Library/Class/MSC/Inc -I../Drivers/BSP/Components -I../Drivers/BSP/STM32F429I_DISCO -I../Drivers/Utilities/Fonts -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-BSP-2f-STM32F429I_DISCO

clean-Drivers-2f-BSP-2f-STM32F429I_DISCO:
	-$(RM) ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery.d ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery.o ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_io.d ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_io.o ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_lcd.d ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_lcd.o ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_sdram.d ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_sdram.o ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_ts.d ./Drivers/BSP/STM32F429I_DISCO/stm32f429i_discovery_ts.o

.PHONY: clean-Drivers-2f-BSP-2f-STM32F429I_DISCO

