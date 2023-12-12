using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

using pred = YOLOv3MLNet.DataStructures;
using YOLOv3MLNet.DataStructures;
using System.Collections.Generic;
using JetBrains.Annotations;


public class IONNX : MonoBehaviour
{
    //ONNX�� ���� ����
    static int CategoriesCount = 403 + 4;

    public NNModel Model;
    private Model m_RunTimeModel; //���� �ҷ����� ����

    public Texture image;

    //����� ������� UI �̹���
    public RawImage image_result;

    pred.YoloV3Prediction predict = null;

    //Temp
    string[] catecories = new string[CategoriesCount];

    // Start is called before the first frame update
    void Start()
    {
        for(int i = 0; i < CategoriesCount; i++)
        {
            catecories[i] = i.ToString();
        }

        //Importing
        m_RunTimeModel = ModelLoader.Load(Model);
        predict = GetComponent<pred.YoloV3Prediction>();

        //�߷� ���� ���� // GPU �۾� ����
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, m_RunTimeModel);

        //#��ǲ ���ڰ��� �ؽ��� ����
        Tensor input = new Tensor(image);
        //���� �Ѱ� �۾� ����
        worker.Execute(input);

        //��� ��� ���� ������
        string[] output = m_RunTimeModel.outputs.ToArray();

        Tensor output_classe = worker.PeekOutput(output[0]);

        Tensor output_boxes = worker.PeekOutput(output[1]);

        //boxes
        Debug.Log(output_boxes.ToReadOnlyArray().Length);

        //classe
        Debug.Log(output_classe.ToReadOnlyArray().Length);


        //setting
        predict.BBoxes = output_boxes.ToReadOnlyArray();
        predict.Classes = output_classe.ToReadOnlyArray();

        //
        IReadOnlyList<YoloV3Result> result;
        result = predict.GetResults(catecories);


        if(result.Count > 0) 
        { 
            Debug.Log("��� ���� ��" + result[0]); 
            foreach (var item in result)
            {
                Debug.Log(item.Label);
            }
        }


        /* None
        Texture result = output.ToRenderTexture();

        Debug.Log(result);
        image_result.texture = result;
        */

        //�Ҵ� ����
        worker.Dispose();
        output_boxes.Dispose();
        output_classe.Dispose();
        input.Dispose();
        //base = "(`boxes` (n:4032, h:1, w:1, c:4), alloc: Unity.Barracuda.DefaultTensorAllocator,
        //onDevice:(GPU:LayerOutput#-1379914384 (n:1, h:8, w:8, c:1024) buffer: UnityEngine.ComputeBuffer created at: ))"
    }

    public static void Prediction(Texture img)
    {

    }
}


