using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class Road_Calori : MonoBehaviour
{
    //���� �������� ��Ʈ�� �ҷ����� ��ũ
    const string langURL = "https://docs.google.com/spreadsheets/d/1vq5bOnVS79-hKQDKxzsCbdrCMc8oVq3R/edit?usp=drive_link&ouid=100305297284869565799&rtpof=true&sd=true";
    string[] row;
    void Start()
    {
        StartCoroutine(GetLangCo());
    }

    IEnumerator GetLangCo()
    {
        UnityWebRequest www = UnityWebRequest.Get(langURL);
        yield return www.SendWebRequest();
        SetRowList(www.downloadHandler.text);
    }

    void SetRowList(string tsv)
    {
        //�ϳ��� �� ������ ���� ������ �и�
        row = tsv.Split(' ');
    }

    public string RoadData(string key)
    {
        Debug.Log(key);
        //�ҷ��� �������� ��Ʈ�� �ε��� ��ȣ �� key�� ���� �ܾ ã�´�
        for (int i = 0; i < row.Length; i++)
        {
            string[] column = row[i].Split(' ');  //���� ������ ���ڿ� ���� column���� ���� ����+1���� ���ڿ��� ����

            if (column[0] == key)
            {  //key�� ���� �ѱ� �ܾ ã�� �ٸ� �׿� ������ �����ϴ� ���ܾ� ��ȯ
                column[0] = column[1].Replace("\r", "");
                return column[0];
            }
        }
        //������ ���� ��ȯ
        Debug.Log("..");
        return "";
    }
}
