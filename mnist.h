#ifndef __Mnist_
#define __Mnist_

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

//�������ݼ��������ݵ�����
typedef struct Signaldata{
	int c;           // ���ݳ�
	int r;           // ���ݿ�Ĭ��Ϊ2
	float** IQData; // IQ���ݶ�̬����
}Signaldata;

typedef struct SignaldataArr{
	int dataNum;        // ���ݵ�����
	Signaldata* DataPtr;  // ����ָ��
}SignaldataArrS, *DataArr; //���ݴ洢

typedef struct DataLabel{
	int l;            // �����ǩ����
	float* LabelData; // �����ǩ����
}MnistLabel;

typedef struct DataLabelArr{
	int LabelNum;
	MnistLabel* LabelPtr;
} MnistLabelArrS, *LabelArr; //�洢���ݱ�ǩ������

LabelArr read_Lable(const char* filename); 

DataArr read_Data(const char* filename);

LabelArr read_Lable_ar();

DataArr read_Data_ar();

void save_Img(DataArr imgarr, char* filedir);

#endif