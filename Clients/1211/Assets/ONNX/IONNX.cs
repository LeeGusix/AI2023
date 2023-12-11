using UnityEngine;
using Unity.Barracuda;
using UnityEngine.UI;

public class IONNX : MonoBehaviour
{
    public NNModel Model;
    private Model m_RunTimeModel; //���� �ҷ����� ����

    public Texture image;

    //����� ������� UI �̹���
    public RawImage image_result;

    // Start is called before the first frame update
    void Start()
    {
        m_RunTimeModel = ModelLoader.Load(Model);

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
        Debug.Log(output_boxes.ToReadOnlyArray());

        Debug.Log(output_classe.ToReadOnlyArray());

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
}
