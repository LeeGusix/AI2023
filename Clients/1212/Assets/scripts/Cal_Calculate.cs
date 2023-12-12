using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class Cal_Calculate : MonoBehaviour
{
    [SerializeField] TMP_InputField user_height, user_weight;
    [SerializeField] TextMeshProUGUI textview;

    public void Calc()
    {
        int height = int.Parse(user_height.text); 
        int weight = int.Parse(user_weight.text);
        int Standard_Weight = 0;
        int Calories_Needed = 0;
        string User_staus;

        //ǥ�� ü�� ���
        if (height < 150)
        {
            Standard_Weight = height - 100;
        }
        else if(height >= 150 && height < 160)
        {
            Standard_Weight = (height - 150) / 2 + 50;
        }
        else
        {
            Standard_Weight = (int)((height - 100) * 0.9);
        }

        if (Standard_Weight < weight)
        {
            User_staus = "ü�� ����";
        }
        else if (Standard_Weight > weight)
        {
            User_staus = "ü�� ����";
        }
        else
        {
            User_staus = "ü�� ����";
        }

        //�ϴ� �ʿ� Į�θ� ���
        Calories_Needed = Standard_Weight * 30 - 35;

        //�ؽ�Ʈ ���
        textview.text = "ǥ�� ü��<color=#11f3af><b>" + Standard_Weight + "</b></color>�Դϴ�.\n<color=#11f3af><b>"
            + User_staus + "</b></color>�� ���� �Ϸ� <color=#11f3af><b>" + Calories_Needed + "</b></color>�� ���� ���밡 ����˴ϴ�.";
    }
}