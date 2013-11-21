//
//  ViewController.m
//  SpecialCamera
//
//  Created by iDreamK on 13-10-27.
//  Copyright (c) 2013年 iDreamK. All rights reserved.
//

#import "ViewController.h"
#import "cv.h"
#import "highgui.h"

@interface ViewController ()

@end

@implementation ViewController

struct MYPOINT
{
    double x;
    double y;
};

int m_nBasePt = 4;

- (void)viewDidLoad
{
    int p[3];
    p[0] = CV_IMWRITE_JPEG_QUALITY;
    p[1] = 10;
    p[2] = 0;
    
    const char* test = "/Users/idreamk/Desktop/SpecialCamera/SpecialCamera/test4.jpg";
    IplImage* I = cvLoadImage(test, CV_LOAD_IMAGE_COLOR);
    IplImage* I0 = cvCreateImage(cvGetSize(I), IPL_DEPTH_8U, 1);
    IplImage* I1 = cvCreateImage(cvGetSize(I), IPL_DEPTH_8U, 3);//temp
    IplImage* I2 = cvCreateImage(cvGetSize(I), IPL_DEPTH_8U, 1);
    CvSeq *lines = 0;
    int binaryThreshold = 0;
    CvMemStorage *storage = cvCreateMemStorage(0);
    CvPoint crossI[20],crossII[20],crossIII[20],crossIV[20];
    CvPoint SrcPointI,SrcPointII,SrcPointIII,SrcPointIV;
        SrcPointI = SrcPointII = SrcPointIII = SrcPointIV = cvPoint(I->width/2, I->height/2);
    
    cvCvtColor(I, I0, CV_BGR2GRAY);//彩色转为灰度
    binaryThreshold = DetectThreshold(I0, 100, 0);//迭代求二值化阈值
    cvThreshold(I0, I0, binaryThreshold, 255, CV_THRESH_OTSU);//灰度转为二值化
    cvSmooth(I0, I0, CV_MEDIAN, 3, 0, 0, 0);//平滑去噪
    
    IplConvKernel* element = cvCreateStructuringElementEx(20, 20, 10, 10, CV_SHAPE_RECT, NULL);//设置腐蚀、膨胀单元
    cvNot(I0, I0);//反色
    cvErode(I0, I0, element, 1);//腐蚀
    cvDilate(I0, I0, element, 3);//膨胀
    cvCanny(I0, I2, binaryThreshold, 255, 3);//取边缘
    
    cvCvtColor(I2, I1, CV_GRAY2RGB);
    
#if 1
    lines = cvHoughLines2(I2, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, 150, 0, 0);//取霍夫直线
    for(int i = 0; i<MIN(lines->total,20); i++)
    {
        float *line = (float*)cvGetSeqElem(lines,i);
        float rho = line[0];
        float threta = line[1];
        CvPoint pt1, pt2;
        double a = cos(threta),b = sin(threta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 5000*(-b));
        pt1.y = cvRound(y0 + 5000*(a));
        pt2.x = cvRound(x0 - 5000*(-b));
        pt2.y = cvRound(y0 - 5000*(a));
        //cvLine(I1,pt1,pt2,CV_RGB(0,255,255),1,100,0);//画霍夫直线
    }
    for(int i = 0; i<MIN(lines->total-1,19); i++)
    {
        for(int j = i+1; j<MIN(lines->total,20); j++)
        {
            float *line1 = (float*)cvGetSeqElem(lines,i);
            float rho1 = line1[0];
            float threta1 = line1[1];
            CvPoint pt1, pt2;
            double a10 = cos(threta1),b10 = sin(threta1);
            double x10 = a10 * rho1, y10 = b10 * rho1;
            pt1.x = cvRound(x10 + 5000*(-b10));
            pt1.y = cvRound(y10 + 5000*(a10));
            pt2.x = cvRound(x10 - 5000*(-b10));
            pt2.y = cvRound(y10 - 5000*(a10));
            
            float *line2 = (float*)cvGetSeqElem(lines,j);
            float rho2 = line2[0];
            float threta2 = line2[1];
            CvPoint pt3, pt4;
            double a20 = cos(threta2),b20 = sin(threta2);
            double x20 = a20 * rho2, y20 = b20 * rho2;
            pt3.x = cvRound(x20 + 5000*(-b20));
            pt3.y = cvRound(y20 + 5000*(a20));
            pt4.x = cvRound(x20 - 5000*(-b20));
            pt4.y = cvRound(y20 - 5000*(a20));
            
            //求直线交点
            CvPoint cross;
            double b1 = (pt2.x*pt1.y - pt1.x*pt2.y)/(double)(pt2.x - pt1.x);
            double k1 = (pt2.y - pt1.y)/(double)(pt2.x - pt1.x);
            double b2 = (pt4.x*pt3.y - pt3.x*pt4.y)/(double)(pt4.x - pt3.x);
            double k2 = (pt4.y - pt3.y)/(double)(pt4.x - pt3.x);
            cross.x = cvRound((b2 - b1)/(double)(k1 - k2));
            cross.y = cvRound((k2*b1 - k1*b2)/(double)(k2 - k1));
            
            if (cross.x<I2->width && cross.x>(I2->width/2) && cross.y>0 && cross.y<(I2->height/2))
            {
                crossI[i] = cross;
                double length1 = (cross.x - (I->width/2))*(cross.x - (I->width/2))
                                 + (cross.y - (I->height/2))*(cross.y - (I->height/2));
                double length2 = (SrcPointI.x - (I->width/2))*(SrcPointI.x - (I->width/2))
                                 + (SrcPointI.y - (I->height/2))*(SrcPointI.y - (I->height/2));
                if (length1>length2) {
                    SrcPointI = cross;
                }
            }
            else if (cross.x<(I2->width/2) && cross.x>0 && cross.y>0 && cross.y<(I2->height/2))
            {
                crossII[i] = cross;
                double length1 = (cross.x - (I->width/2))*(cross.x - (I->width/2))
                                 + (cross.y - (I->height/2))*(cross.y - (I->height/2));
                double length2 = (SrcPointII.x - (I->width/2))*(SrcPointII.x - (I->width/2))
                                 + (SrcPointII.y - (I->height/2))*(SrcPointII.y - (I->height/2));
                if (length1>length2) {
                    SrcPointII = cross;
                }
                
            }
            else if (cross.x<(I2->width/2) && cross.x>0 && cross.y>(I2->height/2) && cross.y<I2->height)
            {
                crossIII[i] = cross;
                double length1 = (cross.x - (I->width/2))*(cross.x - (I->width/2))
                                 + (cross.y - (I->height/2))*(cross.y - (I->height/2));
                double length2 = (SrcPointIII.x - (I->width/2))*(SrcPointIII.x - (I->width/2))
                                 + (SrcPointIII.y - (I->height/2))*(SrcPointIII.y - (I->height/2));
                if (length1>length2) {
                    SrcPointIII = cross;
                }

            }
            else if (cross.x<I2->width && cross.x>(I2->width/2) && cross.y>(I2->height/2) && cross.y<I2->height)
            {
                crossIV[i] = cross;
                double length1 = (cross.x - (I->width/2))*(cross.x - (I->width/2))
                                 + (cross.y - (I->height/2))*(cross.y - (I->height/2));
                double length2 = (SrcPointIV.x - (I->width/2))*(SrcPointIV.x - (I->width/2))
                                 + (SrcPointIV.y - (I->height/2))*(SrcPointIV.y - (I->height/2));
                if (length1>length2) {
                    SrcPointIV = cross;
                }
            }
        }
    }
    //画四个顶点
    cvCircle(I1, SrcPointI, 20, cvScalar(255, 0, 0, 0), 5, 8, 0);
    cvCircle(I1, SrcPointII, 20, cvScalar(255, 0, 0, 0), 10, 8, 0);
    cvCircle(I1, SrcPointIII, 20, cvScalar(255, 0, 0, 0), 10, 8, 0);
    cvCircle(I1, SrcPointIV, 20, cvScalar(255, 0, 0, 0), 10, 8, 0);
#else
    lines = cvHoughLines2(I2, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180, 150, 0, 0);
    for(int i = 0; i<lines->total; ++i)
    {
        CvPoint *line = (CvPoint*)cvGetSeqElem(lines,i);  
        cvLine(I1,line[0],line[1],CV_RGB(255,0,255),1,100,0);
    }  
#endif
    UIImage *t = [self UIImageFromIplImage:I1];
    
    float sizePara = 0.85;
    CvPoint pBasePoints[4];
    CvPoint pSrcPoints[4];
    
    CvPoint BasePointI = cvPoint(cvRound(sizePara*I->width), 0);
    CvPoint BasePointII = cvPoint(0, 0);
    CvPoint BasePointIII = cvPoint(0, cvRound(sizePara*I->height));
    CvPoint BasePointIV = cvPoint(cvRound(sizePara*I->width), cvRound(sizePara*I->height));
    pBasePoints[0] = BasePointII;
    pBasePoints[1] = BasePointI;
    pBasePoints[2] = BasePointIV;
    pBasePoints[3] = BasePointIII;
    pSrcPoints[0] = SrcPointII;
    pSrcPoints[1] = SrcPointI;
    pSrcPoints[2] = SrcPointIV;
    pSrcPoints[3] = SrcPointIII;
    
    //ImProjRestore(I1, pBasePoints, pSrcPoints, 1);
    
    NSString *aPath=[NSString stringWithFormat:@"/Users/idreamk/Desktop/SpecialCamera/SpecialCamera/%@.jpg",@"test0"];
    NSData *imgData = UIImageJPEGRepresentation(t,0);
    [imgData writeToFile:aPath atomically:YES];

    [super viewDidLoad];

}

//迭代法求灰度图转二值图的阈值
int DetectThreshold(const IplImage *img, int nMaxIter, int iDiffRec)
{
    //图像信息
    int iHeight = img->height;
    int iWidth = img->width;
    int iStep = img->widthStep/sizeof(uchar);
    uchar *pData = (uchar*)img->imageData;
    
    iDiffRec =0;
    int F[256]={ 0 }; //直方图数组
    int iTotalGray=0;//灰度值和
    int iTotalPixel =0;//像素数和
    Byte bt;//某点的像素值
    
    uchar iThrehold = 0; //阀值
    uchar iNewThrehold = 0; //新阀值
    uchar iMaxGrayValue=0,iMinGrayValue=255;//原图像中的最大灰度值和最小灰度值
    uchar iMeanGrayValue1,iMeanGrayValue2;
    
    //获取(i,j)的值，存于直方图数组F
    for(int i=0;i<iWidth;i++)
    {
        for(int j=0;j<iHeight;j++)
        {
            bt = pData[i*iStep+j];
            if(bt<iMinGrayValue)
                iMinGrayValue = bt;
            if(bt>iMaxGrayValue)
                iMaxGrayValue = bt;
            F[bt]++;
        }
    }
    
    iThrehold =0;
    iNewThrehold = (iMinGrayValue+iMaxGrayValue)/2;//初始阀值(图像的平均灰度)
    iDiffRec = iMaxGrayValue - iMinGrayValue;
    
    for(int a=0;(abs(iThrehold-iNewThrehold)>0.5)&&a<nMaxIter;a++)//迭代中止条件
    {
        iThrehold = iNewThrehold;
        //小于当前阀值部分的平均灰度值
        for(int i=iMinGrayValue;i<iThrehold;i++)
        {
            iTotalGray += F[i]*i;//F[]存储图像信息
            iTotalPixel += F[i];
        }
        
        iMeanGrayValue1 = (uchar)(iTotalGray/iTotalPixel);
        //大于当前阀值部分的平均灰度值
        iTotalPixel =0;
        iTotalGray =0;
        for(int j=iThrehold+1;j<iMaxGrayValue;j++)
        {
            iTotalGray += F[j]*j;//F[]存储图像信息
            iTotalPixel += F[j];
        }
        
        iMeanGrayValue2 = (uchar)(iTotalGray/iTotalPixel);
        
        iNewThrehold = (iMeanGrayValue2+iMeanGrayValue1)/2;        //新阀值
        iDiffRec = abs(iMeanGrayValue2 - iMeanGrayValue1);
    }
    
    return iThrehold;
}

//IplImage图像转为UIImage的图像
// NOTE You should convert color mode as RGB before passing to this function
- (UIImage *)UIImageFromIplImage:(IplImage *)image
{
	NSLog(@"IplImage (%d, %d) %d bits by %d channels, %d bytes/row %s", image->width, image->height, image->depth, image->nChannels, image->widthStep, image->channelSeq);
    
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	NSData *data = [NSData dataWithBytes:image->imageData length:image->imageSize];
	CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
	CGImageRef imageRef = CGImageCreate(image->width, image->height,
										image->depth, image->depth * image->nChannels, image->widthStep,
										colorSpace, kCGImageAlphaNone|kCGBitmapByteOrderDefault,
										provider, NULL, false, kCGRenderingIntentDefault);
	UIImage *ret = [UIImage imageWithCGImage:imageRef];
	CGImageRelease(imageRef);
	CGDataProviderRelease(provider);
	CGColorSpaceRelease(colorSpace);
	return ret;
}

- (void)didReceiveMemoryWarning
{

    [super didReceiveMemoryWarning];

}


/*******************
 void CImgProcess::GetProjPara(CPoint* pPointBase, CPoint* pPointSampl, double* pDbProjPara)
 
 功能：根据基准点对儿（4对儿）确定变换参数
 
 参数：
 CPoint* pPointBase：基准图像的基准点
 CPoint* pPointSampl：输入图像的基准点
 double* pDbProjPara：变换参数
 
 返回值:
 无
 
 *******************/
void GetProjPara(CvPoint* pPointBase, CvPoint* pPointSampl, double* pDbProjPara)
{
	int i;
    
	//投影线性方程的系数矩阵
	double **ppParaMat = (double**)malloc(sizeof(double*)*(m_nBasePt));
	for(i=0; i<m_nBasePt; i++)
	{
		ppParaMat[i] = (double*)malloc(sizeof(double)*(m_nBasePt));
	}
    
	for(i=0; i<m_nBasePt; i++)
	{
		ppParaMat[i][0] = pPointBase[i].x;
		ppParaMat[i][1] = pPointBase[i].y;
		ppParaMat[i][2] = pPointBase[i].x * pPointBase[i].y;
		ppParaMat[i][3] = 1;
	}
    
	double* pResMat;//结果矩阵
	pResMat = (double*)malloc(sizeof(double)*(m_nBasePt));
	for(i=0; i<m_nBasePt; i++)//计算前四个系数 c1,c2,c3,c4
	{
		pResMat[i] = pPointSampl[i].x; //投影线性方程的值x'
	}
	
	// 采用左乘系数矩阵的逆矩阵的方法解出投影变换的前4个系数 c1,c2,c3,c4
	InvMat(ppParaMat, m_nBasePt);
	ProdMat(ppParaMat, pResMat, pDbProjPara, m_nBasePt, 1, m_nBasePt);//求出前4个系数
	
	for(i=0; i<m_nBasePt; i++)//计算后四个系数 c5,c6,c7,c8
	{
		pResMat[i] = pPointSampl[i].y; //投影线性方程的值y'
	}
	// 采用左乘系数矩阵的逆矩阵的方法解出投影变换的后4个系数 c5,c6,c7,c8
	ProdMat(ppParaMat, pResMat, pDbProjPara+m_nBasePt, m_nBasePt, 1, m_nBasePt);//求出后4个系数
    
    
	//释放空间
	free(pResMat);
    
	for(i=0; i<m_nBasePt; i++)
	{
		free(ppParaMat[i]);
	}
	free(ppParaMat);
}



/*******************
 BOOL CImgProcess::InvMat(double** ppDbMat, int nLen)
 
 功能：计算ppDbMat的逆矩阵
 
 注：ppDbMat必须为方阵
 
 参数：
 double** ppDbMat：输入矩阵
 int nLen：矩阵ppDbMat的尺寸
 
 返回值:
 BOOL
 =true：执行成功
 =false：计算过程中出现错误
 *******************/
BOOL InvMat(double** ppDbMat, int nLen)
{
	double* pDbSrc = (double*)malloc(sizeof(double)*(nLen * nLen));
	
	int *is,*js,i,j,k;
    
	//保存要求逆的输入矩阵
	int nCnt = 0;
	for(i=0; i<nLen; i++)
	{
		for(j=0; j<nLen; j++)
			pDbSrc[nCnt++] = ppDbMat[i][j];
	}
    
	double d,p;
	is = (int*)malloc(sizeof(int)*(nLen));
	js = (int*)malloc(sizeof(int)*(nLen));
	for(k=0;k<nLen;k++)
	{
		d=0.0;
		for(i=k;i<nLen;i++)
			for(j=k;j<nLen;j++)
			{
				p=fabs(pDbSrc[i*nLen + j]); //找到绝对值最大的系数
				if(p>d)
				{
					d = p;
                    
					//记录绝对值最大的系数的行、列索引
					is[k] = i;
					js[k] = j;
				}
			}
		if(d+1.0==1.0)
		{//系数全是0，系数矩阵为0 阵，此时为奇异矩阵
			free(is);
			free(js);
			return FALSE;
		}
		if(is[k] != k) //当前行不包含最大元素
			for(j=0;j<nLen;j++)
			{
				//交换两行元素
				p = pDbSrc[k*nLen + j];
				pDbSrc[k*nLen + j] = pDbSrc[(is[k]*nLen) + j];
				pDbSrc[(is[k])*nLen + j] = p;
			}
		if(js[k] != k) //当前列不包含最大元素
			for(i=0; i<nLen; i++)
			{
				//交换两列元素
				p = pDbSrc[i*nLen + k];
				pDbSrc[i*nLen + k] = pDbSrc[i*nLen + (js[k])];
				pDbSrc[i*nLen + (js[k])] = p;
			}
        
		pDbSrc[k*nLen + k]=1.0/pDbSrc[k*nLen + k]; //求主元的倒数
		
		// a[k,j]a[k,k] -> a[k,j]
		for(j=0; j<nLen; j++)
			if(j != k)
			{
				pDbSrc[k*nLen + j]*=pDbSrc[k*nLen + k];
			}
        
		// a[i,j] - a[i,k]a[k,j] -> a[i,j]
		for(i=0; i<nLen; i++)
			if(i != k)
				for(j=0; j<nLen; j++)
					if(j!=k)
					{
						pDbSrc[i*nLen + j] -= pDbSrc[i*nLen + k]*pDbSrc[k*nLen + j];
					}
        
		// -a[i,k]a[k,k] -> a[i,k]
		for(i=0; i<nLen; i++)
			if(i != k)
			{
				pDbSrc[i*nLen + k] *= -pDbSrc[k*nLen + k];
			}
	}
	for(k=nLen-1; k>=0; k--)
	{
		//恢复列
		if(js[k] != k)
			for(j=0; j<nLen; j++)
			{
				p = pDbSrc[k*nLen + j];
				pDbSrc[k*nLen + j] = pDbSrc[(js[k])*nLen + j];
				pDbSrc[(js[k])*nLen + j] = p;
			}
		//恢复行
		if(is[k] != k)
			for(i=0; i<nLen; i++)
			{
				p = pDbSrc[i*nLen + k];
				pDbSrc[i*nLen + k] = pDbSrc[i*nLen +(is[k])];
				pDbSrc[i*nLen + (is[k])] = p;
			}
	}
    
	//将结果拷贝回系数矩阵ppDbMat
	nCnt = 0;
	for(i=0; i<m_nBasePt; i++)
	{
		for(j=0; j<m_nBasePt; j++)
		{
			ppDbMat[i][j] = pDbSrc[nCnt++];
		}
	}
    
	//释放空间
	free(is);
	free(js);
	free(pDbSrc);
    
	return TRUE;
    
}



/*******************
 void CImgProcess::ProdMat(double** ppDbMat, double *pDbSrc2, double *pDbDest, int y, int x, int z)
 
 功能：计算两矩阵的乘积
 
 注：该函数计算两个矩阵的相乘，然后将相乘的结果存放在pDbDest中。
 其中pDbSrc1 *的大小为 x*z，pDbSrc2的大小为 z*y，pDbDest的大小为 x*y
 
 参数：
 double  *pDbSrc1	- 指向相乘矩阵1的内存
 double  *pDbSrc2	- 指向相乘矩阵2的内存
 double  *pDbDest   - 存放矩阵相乘运行结果的内存指针
 int     x		- 矩阵的尺寸，具体参见函数注
 int     y		- 矩阵的尺寸，具体参见函数注
 int     z		- 矩阵的尺寸，具体参见函数注
 
 返回值:
 无
 
 *******************/
void ProdMat(double** ppDbMat, double *pDbSrc2, double *pDbDest, int y, int x, int z)
{
	int nCnt = 0;
	int i,j;
	double * pDbSrc1 = (double*)malloc(sizeof(double)*(m_nBasePt * m_nBasePt));
	for(i=0; i<m_nBasePt; i++)
	{
		for(j=0; j<m_nBasePt; j++)
			pDbSrc1[nCnt++] = ppDbMat[i][j];
	}
    
	for(i=0;i<y;i++)
	{
		for(j=0;j<x;j++)
		{
			pDbDest[i*x + j] = 0;
			for(int m=0;m<z;m++)
				pDbDest[i*x + j] += pDbSrc1[i*z + m]*pDbSrc2[m*x + j];
		}
	}
    
	nCnt = 0;
	for(i=0; i<m_nBasePt; i++)
	{
		for(j=0; j<m_nBasePt; j++)
			ppDbMat[i][j] = pDbSrc1[nCnt++];
	}
    
	free(pDbSrc1);
}




/*******************
 MYPOINT CImgProcess::ProjTrans(CPoint pt, double* pDbProjPara)
 
 功能：根据变换参数对点pt实施投影变换
 
 参数：
 CPoint pt：要进行投影变换的点坐标
 double* pDbProjPara：变换参数
 
 返回值:
 MYPOINT
 *******************/
struct MYPOINT ProjTrans(CvPoint pt, double* pDbProjPara)
{
	struct MYPOINT retPt;
	retPt.x = pDbProjPara[0] * pt.x + pDbProjPara[1] * pt.y + pDbProjPara[2] * pt.x * pt.y + pDbProjPara[3];
	retPt.y = pDbProjPara[4] * pt.x + pDbProjPara[5] * pt.y + pDbProjPara[6] * pt.x * pt.y + pDbProjPara[7];
	return retPt;
}




/*******************
 BOOL CImgProcess::ImProjRestore(CImgProcess* pTo, CPoint *pPointBase, CPoint *pPointSampl, bool bInterp)
 
 功能：实施投影变形校正
 
 参数：
 CImgProcess* pTo：校准后图像的 CImgProcess 指针
 CPoint *pPointBase：基准图像的基准点数组
 CPoint *pPointSampl：输入图像的基准点数组
 bool bInterp：是否使用(双线性)插值
 
 返回值:
 MYPOINT
 *******************/
BOOL ImProjRestore(IplImage *pTo, CvPoint *pPointBase, CvPoint *pPointSampl, bool bInterp)
{
	double* pDbProjPara = (double*)malloc(sizeof(double)*(m_nBasePt * 2));
	GetProjPara(pPointBase, pPointSampl, pDbProjPara);
    
	//用得到的变换系数对图像实施变换
	int i, j;
	//pTo = cvSetZero(0); //清空目标图像
    CvMat matheader;
    CvMat *pTO = cvGetMat(pTo, &matheader, NULL, 0);
	int nHeight = pTo->height;
	int nWidth = pTo->width;
	for(i=0; i<nHeight; i++)
	{
		for(j=0; j<nWidth; j++)
		{
			//对每个点(j, i)，计算其投影失真后的点ptProj
			struct MYPOINT ptProj = ProjTrans(cvPoint(j, i), pDbProjPara );
			
			
			if(bInterp)
			{
				int nGray = InterpBilinear(pTo, ptProj.x, ptProj.y); //输入图像（投影变形图像）的对应点灰度
				if(nGray >= 0)
					pTo->SetPixel(j, i, CV_RGB(nGray, nGray, nGray));
				else
					pTo->SetPixel(j, i, CV_RGB(255, 255, 255)); //超出图像范围，填充白色
			}
			else
			{
				int ii = ptProj.y + 0.5; //四舍五入的最近邻插值
				int jj = ptProj.x + 0.5;
				if( ii>0 && ii<pTo->height && jj>0 && jj<pTo->width )
					pTo->SetPixel(j, i, GetPixel(jj, ii));
				else
					pTo->SetPixel(j, i, CV_RGB(255, 255, 255)); //超出图像范围，填充白色
			}
		}
	}
    
	free (pDbProjPara);
	return TRUE;
}

/*******************
 int CImgProcess::InterpBilinear(double x, double y)
 功能：
 双线性插值
 
 参数：
 double x：需要计算插值的横坐标
 double y：需要计算插值的纵坐标
 返回值:
 int 插值的结果
 *******************/
int InterpBilinear(IplImage *pTo, double x, double y)
{
    int cc;
	if((int)(y)==300)
		cc = 1;
	
	// 四个最临近象素的坐标(i1, j1), (i2, j1), (i1, j2), (i2, j2)
	int	x1, x2;
	int	y1, y2;
	
	// 四个最临近象素值
	unsigned char	f1, f2, f3, f4;
	
	// 二个插值中间值
	unsigned char	f12, f34;
	
	double	epsilon = 0.0001;
	
	// 计算四个最临近象素的坐标
	x1 = (int) x;
	x2 = x1 + 1;
	y1 = (int) y;
	y2 = y1 + 1;
	
    int step = pTo->widthStep/sizeof(uchar);
    
	int nHeight = pTo->height;
	int nWidth = pTo->height;
	if( (x < 0) || (x > nWidth - 1) || (y < 0) || (y > nHeight - 1))
	{
		// 要计算的点不在源图范围内，返回-1
		return -1;
	}
	else
	{
		if (fabs(x - nWidth + 1) <= epsilon)
		{
			// 要计算的点在图像右边缘上
			if (fabs(y - nHeight + 1) <= epsilon)
			{
				// 要计算的点正好是图像最右下角那一个象素，直接返回该点象素值
				f1 = (unsigned char)pTo->imageData[y1*step+x1];//GetGray( x1, y1 );
				return f1;
			}
			else
			{
				// 在图像右边缘上且不是最后一点，直接一次插值即可
				f1 = (unsigned char)pTo->imageData[y1*step+x1];//GetGray( x1, y1 );
				f3 = (unsigned char)pTo->imageData[y2*step+x1];//GetGray( x1, y2 );
                
				// 返回插值结果
				return ((int) (f1 + (y -y1) * (f3 - f1)));
			}
		}
		else if (fabs(y - nHeight + 1) <= epsilon)
		{
			// 要计算的点在图像下边缘上且不是最后一点，直接一次插值即可
			f1 = (unsigned char)pTo->imageData[y1*step+x1];//GetGray( x1, y1 );
			f2 = (unsigned char)pTo->imageData[y1*step+x2];//GetGray( x2, y1 );
			
			// 返回插值结果
			return ((int) (f1 + (x -x1) * (f2 - f1)));
		}
		else
		{
			// 计算四个最临近象素值
			f1 = (unsigned char)pTo->imageData[y1*step+x1];//GetGray( x1, y1 );
			f2 = (unsigned char)pTo->imageData[y1*step+x2];//GetGray( x2, y1 );
			f3 = (unsigned char)pTo->imageData[y2*step+x1];//GetGray( x1, y2 );
			f4 = (unsigned char)pTo->imageData[y2*step+x2];//GetGray( x2, y2 );
			
			// 插值1
			f12 = (unsigned char) (f1 + (x - x1) * (f2 - f1));
			
			// 插值2
			f34 = (unsigned char) (f3 + (x - x1) * (f4 - f3));
			
			// 插值3
			return ((int) (f12 + (y -y1) * (f34 - f12)));
		}
	}
}

/**************************************************
 inline BYTE CImg::GetGray(int x, int y)
 
 功能：
 返回指定坐标位置像素的灰度值
 
 参数：
 int x, int y
 指定的像素横、纵坐标值
 返回值：
 给定像素位置的灰度值
 ***************************************************/
//inline Byte GetGray(int x, int y)
//{
//	COLORREF ref = GetPixel(x, y);
//	Byte r, g, b, byte;
//	r = GetRValue(ref);
//	g = GetGValue(ref);
//	b = GetBValue(ref);
//    
//	if(r == g && r == b)
//		return r;
//    
//	double dGray = (0.30*r + 0.59*g + 0.11*b);
//    
//	// 灰度化
//	byte =  (int)dGray;
//
//	return byte;
//}

@end
