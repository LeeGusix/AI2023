using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Exit : MonoBehaviour
{
    public void Continue()
    {
        Debug.Log("����մϴ�.");
        SceneManager.LoadScene("Photo");
    }

    public void End()
    {
        Debug.Log("�����մϴ�.");
        Application.Quit();
    }
}
