#ifndef INC_GLOBAL_H_
#define INC_GLOBAL_H_

#include "main.h"
#include "crc.h"
#include "dma2d.h"
#include "fatfs.h"
#include "i2c.h"
#include "libjpeg.h"
#include "ltdc.h"
#include "spi.h"
#include "usart.h"
#include "usb_host.h"
#include "gpio.h"
#include "fmc.h"
#include "app_x-cube-ai.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "stm32f429i_discovery.h"
#include "stm32f429i_discovery_lcd.h"
#include "stm32f429i_discovery_sdram.h"

// set the number of images in the dataset for test mode
#define IMAGE_NUMBER 50


typedef struct RGB
{
  uint8_t B;
  uint8_t G;
  uint8_t R;
}RGB_typedef;

/* Exported constants --------------------------------------------------------*/
#define LCD_BUFFER   0xD0000000

extern uint32_t offset;
extern RGB_typedef *RGB_matrix;
extern uint8_t _aucLine[2048] __attribute__((section(".ccmram")));
extern ApplicationTypeDef Appli_state;
extern uint8_t resize_image_buffr[32*32*3] __attribute__((section(".ccmram")));
extern __attribute__((section(".ccmram"))) float predictionval[10];
extern __attribute__((section(".ccmram"))) uint8_t class_name_index[10];

void RGB24_to_Float_Asym(uint8_t *pSrc, uint8_t *pDst, uint32_t pixels);
void AI_Output_Display(uint8_t* AI_out_data);

#endif /* INC_GLOBAL_H_ */
