using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

public class Exit : MonoBehaviour
{
    public void Continue()
    {
        Debug.Log("계속합니다.");
        SceneManager.LoadScene("Photo");
    }

    public void End()
    {
        Debug.Log("종료합니다.");
        Application.Quit();
    }
}
