using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class CanvasManager : MonoBehaviour
{
    public GameObject PhotoCanvas;
    public GameObject ResultCanvas;
    public GameObject ExitNotice;

    void Start()
    {
        PhotoCanvas.gameObject.SetActive(true);
        ResultCanvas.gameObject.SetActive(false);
        ExitNotice.gameObject.SetActive(false);
    }

    public void OnNextButtonClicked()
    {
        if (PhotoCanvas.activeSelf == true)
        {
            PhotoCanvas.SetActive(false);
            ResultCanvas.SetActive(true);
        }
        else if (ResultCanvas.activeSelf == true)
        {
            PhotoCanvas.SetActive(false);
            ExitNotice.SetActive(true);
        }
    }
}
