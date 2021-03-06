#ifdef __cplusplus
 extern "C" {
#endif
/**
  ******************************************************************************
  * @file           : app_x-cube-ai.c
  * @brief          : AI program body
  ******************************************************************************
  * This notice applies to any and all portions of this file
  * that are not between comment pairs USER CODE BEGIN and
  * USER CODE END. Other portions of this file, whether
  * inserted by the user or by software development tools
  * are owned by their respective copyright owners.
  *
  * Copyright (c) 2018 STMicroelectronics International N.V.
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted, provided that the following conditions are met:
  *
  * 1. Redistribution of source code must retain the above copyright notice,
  *    this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  *    this list of conditions and the following disclaimer in the documentation
  *    and/or other materials provided with the distribution.
  * 3. Neither the name of STMicroelectronics nor the names of other
  *    contributors to this software may be used to endorse or promote products
  *    derived from this software without specific written permission.
  * 4. This software, including modifications and/or derivative works of this
  *    software, must execute solely and exclusively on microcontroller or
  *    microprocessor devices manufactured by or for STMicroelectronics.
  * 5. Redistribution and use of this software other than as permitted under
  *    this license is void and will automatically terminate your rights under
  *    this license.
  *
  * THIS SOFTWARE IS PROVIDED BY STMICROELECTRONICS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT
  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
  * PARTICULAR PURPOSE AND NON-INFRINGEMENT OF THIRD PARTY INTELLECTUAL PROPERTY
  * RIGHTS ARE DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW. IN NO EVENT
  * SHALL STMICROELECTRONICS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
  * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */
/* Includes ------------------------------------------------------------------*/
#include <string.h>
#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"

/* USER CODE BEGIN includes */
#include "global.h"

/* USER CODE END includes */
/* USER CODE BEGIN initandrun */
#include <stdlib.h>

/* Global handle to reference the instance of the NN */
AI_ALIGNED(4)
static ai_i8 in_data[AI_NETWORK_IN_1_SIZE_BYTES];

AI_ALIGNED(4)
static ai_i8 out_data[AI_NETWORK_OUT_1_SIZE_BYTES];
static ai_handle network = AI_HANDLE_NULL ;
static ai_buffer ai_input[AI_NETWORK_IN_NUM] = AI_NETWORK_IN ;
static ai_buffer ai_output[AI_NETWORK_OUT_NUM] = AI_NETWORK_OUT ;
const char* cifar10_label[] = {"Plane:", "Car:  ", "Bird: ", "Cat:  ", "Deer: ", "Dog:  ", "Frog: ", "Horse:", "Ship: ", "Truck:"};
__attribute__((section(".ccmram"))) char msg[70];
uint32_t nn_inference_time;
uint32_t tot_inference_time;
int right_image;
int image_number;
float accuracy;

/*
 * Init function to create and initialize a NN.
 */
int aiInit(const ai_u8* activations)
{
    ai_error err;

    /* 1 - Specific AI data structure to provide the references of the
     * activation/working memory chunk and the weights/bias parameters */
    const ai_network_params params = {
            AI_NETWORK_DATA_WEIGHTS(ai_network_data_weights_get()),
            AI_NETWORK_DATA_ACTIVATIONS(activations)
    };

    /* 2 - Create an instance of the NN */
    err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);

    if (err.type != AI_ERROR_NONE) {
	    return -1;
    }

    /* 3 - Initialize the NN - Ready to be used */
    if (!ai_network_init(network, &params)) {
        err = ai_network_get_error(network);
        ai_network_destroy(network);
        network = AI_HANDLE_NULL;
	    return -2;
    }
    return 0;
}

/*
 * Run function to execute an inference.
 */
int aiRun(const void *in_data, void *out_data)
{
    ai_i32 nbatch;
    ai_error err;

    /* Parameters checking */
    if (!in_data || !out_data || !network)
        return -1;

    /* Initialize input/output buffer handlers */
    ai_input[0].n_batches = 1;
    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_output[0].n_batches = 1;
    ai_output[0].data = AI_HANDLE_PTR(out_data);

    /* 2 - Perform the inference */
    nbatch = ai_network_run(network, &ai_input[0], &ai_output[0]);

    if (nbatch != 1) {
        err = ai_network_get_error(network);
        // ...
        return err.code;
    }
    return 0;
}
/* USER CODE END initandrun */

/*************************************************************************
  *
  */
void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 0 */
    /* Activation/working buffer is allocated as a static memory chunk
     * (bss section) */
    AI_ALIGNED(4)
    static ai_u8 activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

    aiInit(activations);
    /* USER CODE END 0 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 1 */
    int res;
    int16_t i,x=11;

    printf("Demo\r\n");
    BSP_LCD_DisplayStringAtLine(19,(uint8_t*)"Demo");
    /* Perform the inference */
    RGB24_to_Float_Asym(&resize_image_buffr[0], (uint8_t*)&in_data[0], 32* 32);
    res = aiRun(in_data, out_data);

    if (res) {
        printf("AI error %d\r\n",res);
        BSP_LCD_DisplayStringAtLine(19,(uint8_t*)"AI error");
        return;
    }
    printf("Display result\r\n");
    AI_Output_Display((uint8_t*)out_data);
    BSP_LCD_DisplayStringAtLine(9,(uint8_t*)"Prediction:");

    for(i=9;i>=7;i--){
        if(predictionval[i]>1){
            sprintf(msg,"%s %.2f%%",cifar10_label[class_name_index[i]],predictionval[i]);
            printf("Pred. %s Confi. %.2f%%\r\n",cifar10_label[class_name_index[i]],predictionval[i]);
            BSP_LCD_DisplayStringAtLine(x++,(uint8_t*)msg);
        }
    }
    /* USER CODE END 1 */
}

void test(int class_index)
{
    int res;
    uint32_t Tinf1;
    uint32_t Tinf2;

    printf("Test\r\n");
    BSP_LCD_DisplayStringAtLine(19,(uint8_t*)"Test");
    /* Perform the inference */
    RGB24_to_Float_Asym(&resize_image_buffr[0], (uint8_t*)&in_data[0], 32* 32);
    Tinf1 = HAL_GetTick();
    res = aiRun(in_data, out_data);
    Tinf2 = HAL_GetTick();
    nn_inference_time = ((Tinf2>Tinf1)?(Tinf2-Tinf1):((1<<24)-Tinf1+Tinf2));
    tot_inference_time = tot_inference_time + nn_inference_time;

    if (res) {
        printf("AI error %d\r\n",res);
        BSP_LCD_DisplayStringAtLine(19,(uint8_t*)"AI error");
        return;
    }
    printf("Display result\r\n");
    AI_Output_Display((uint8_t*)out_data);

    if (class_name_index[9]==class_index){
        right_image=right_image +1;
    }
    image_number = image_number + 1;

    if (image_number==IMAGE_NUMBER){
        sprintf(msg, "Image: %d/%d", image_number, IMAGE_NUMBER);
        BSP_LCD_DisplayStringAtLine(0,(uint8_t*)msg);

        accuracy = (float)right_image/image_number*100;

        if (accuracy==accuracy){
            sprintf(msg, "Accuracy: %.2f%%  ", accuracy);
            BSP_LCD_DisplayStringAtLine(2,(uint8_t*)msg);
        }

        sprintf(msg, "Inference: %ldms", nn_inference_time);
        BSP_LCD_DisplayStringAtLine(4,(uint8_t*)msg);

        HAL_Delay(2000);

        sprintf(msg, "Total inf.: %.2fs", (float)tot_inference_time/1000);
        BSP_LCD_DisplayStringAtLine(4,(uint8_t*)msg);

        printf("Test completed\r\n");
        BSP_LCD_DisplayStringAtLine(19,(uint8_t*)"Test completed");
    }
    else {
        sprintf(msg, "Image: %d/%d", image_number, IMAGE_NUMBER);
        BSP_LCD_DisplayStringAtLine(0,(uint8_t*)msg);

        accuracy = (float)right_image/image_number*100;
        
        if (accuracy==accuracy){
            sprintf(msg, "Accuracy: %.2f%%  ", accuracy);
            BSP_LCD_DisplayStringAtLine(2,(uint8_t*)msg);
        }

        sprintf(msg, "Inference: %ldms", nn_inference_time);
        BSP_LCD_DisplayStringAtLine(4,(uint8_t*)msg);
    }
}

#ifdef __cplusplus
}
#endif
