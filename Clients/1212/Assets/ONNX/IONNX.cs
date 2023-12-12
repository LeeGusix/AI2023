using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

public class IONNX : MonoBehaviour
{
    public NNModel Model;
    private Model m_RunTimeModel; //모델을 불러오기 위함

    public Texture image;

    //결과를 출력해줄 UI 이미지
    public RawImage image_result;

    // Start is called before the first frame update
    void Start()
    {
        m_RunTimeModel = ModelLoader.Load(Model);

        //추론 엔진 생성 // GPU 작업 예약
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, m_RunTimeModel);

        //#인풋 인자값인 텍스쳐 지정
        Tensor input = new Tensor(image);
        //값을 넘겨 작업 실행
        worker.Execute(input);

        //출력 결과 값을 도출함
        string[] output = m_RunTimeModel.outputs.ToArray();

        Tensor output_classe = worker.PeekOutput(output[0]);

        Tensor output_boxes = worker.PeekOutput(output[1]);

        //boxes
        Debug.Log(output_boxes.ToReadOnlyArray());

        Debug.Log(output_classe.ToReadOnlyArray());

        /* None
        Texture result = output.ToRenderTexture();

        Debug.Log(result);
        image_result.texture = result;
        */

        //할당 해제
        worker.Dispose();
        output_boxes.Dispose();
        output_classe.Dispose();
        input.Dispose();
        //base = "(`boxes` (n:4032, h:1, w:1, c:4), alloc: Unity.Barracuda.DefaultTensorAllocator,
        //onDevice:(GPU:LayerOutput#-1379914384 (n:1, h:8, w:8, c:1024) buffer: UnityEngine.ComputeBuffer created at: ))"
    }
}
