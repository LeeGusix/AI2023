using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

using pred = YOLOv3MLNet.DataStructures;
using YOLOv3MLNet.DataStructures;
using System.Collections.Generic;
using JetBrains.Annotations;


public class IONNX : MonoBehaviour
{
    //ONNX의 값을 참고
    static int CategoriesCount = 403 + 4;

    //바라쿠다 모델
    public NNModel Model;
    private Model m_RunTimeModel; //모델을 불러오기 위함

    //임시 분류 이미지
    public Texture2D image;

    //결과를 출력해줄 UI 이미지
    public RawImage image_result;

    //결과값 도출 컴포넌트ㄴ
    YoloV3Prediction predict = null;

    //임시 카테고리
    string[] catecories = new string[CategoriesCount];

    // Start is called before the first frame update
    void Awake()
    {
        for(int i = 0; i < CategoriesCount; i++)
        {
            catecories[i] = i.ToString();
        }

        //Importing
        m_RunTimeModel = ModelLoader.Load(Model);
        predict = GetComponent<pred.YoloV3Prediction>();
    }

    private void Start()
    {
        //test
        Prediction(image);
    }

    public IReadOnlyList<YoloV3Result> Prediction(Texture2D img)
    {
        //추론 엔진 생성 // GPU 작업 예약
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, m_RunTimeModel);

        //Texture2D texture = img;// ScaleTexture(img, predict.ImageWidth, predict.ImageHeight);
        //texture.Reinitialize(256, 256);


        //#인풋 인자값인 텍스쳐 지정
        TensorShape shape = new TensorShape(1,256,256,3);
        Tensor input = new Tensor(img);//.Reshape(shape);
        Debug.Log("Shape : "+input.shape + "Length : " +input.length );
        //값을 넘겨 작업 실행
        worker.Execute(input);

        //출력 결과 값을 도출함
        string[] output = m_RunTimeModel.outputs.ToArray();

        Tensor output_classe = worker.PeekOutput(output[0]);

        Tensor output_boxes = worker.PeekOutput(output[1]);

        //boxes
        Debug.Log($"BOXES : {output_boxes.ToReadOnlyArray().Length}");

        //classe
        Debug.Log($"CLASSE : {output_classe.ToReadOnlyArray().Length}");

        //setting
        predict.BBoxes = output_boxes.ToReadOnlyArray();
        predict.Classes = output_classe.ToReadOnlyArray();
        predict.ImageWidth = img.width;
        predict.ImageHeight = img.height;

        Debug.Log($"BOXES.COUNT : {predict.BBoxes.Length} / Classes.COUNT : {predict.Classes.Length} / ImageWidth: {predict.ImageWidth} / ImageHeight : {predict.ImageHeight}");

        //결과값을 가져옵니다.
        IReadOnlyList<YoloV3Result> result;
        result = predict.GetResults(catecories);

        if (result.Count > 0)
        {
            Debug.Log("도출된 결과 개수 : " + result.Count);
            foreach (var item in result)
            {
                Debug.Log("라벨 : " + item.Label);
            }
        }

        //종료
        worker.Dispose();
        output_boxes.Dispose();
        output_classe.Dispose();
        input.Dispose();

        predict.BBoxes = null;
        predict.Classes = null;

        return result;
    }

    private Texture2D ScaleTexture(Texture2D source, float targetWidth, float targetHeight)
    {
        Texture2D result = new Texture2D((int)targetWidth, (int)targetHeight, source.format, true);
        Color[] rpixels = result.GetPixels(0);
        float incX = (1.0f / (float)targetWidth);
        float incY = (1.0f / (float)targetHeight);
        for (int px = 0; px < rpixels.Length; px++)
        {
            rpixels[px] = source.GetPixelBilinear(incX * ((float)px % targetWidth), incY * ((float)Mathf.Floor(px / targetWidth)));
        }
        result.SetPixels(rpixels, 0);
        result.Apply();

        return result;
    }
}


