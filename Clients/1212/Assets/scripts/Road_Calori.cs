using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class Road_Calori : MonoBehaviour
{
    //번역 스프리드 시트를 불러오는 링크
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
        //하나의 긴 문장을 공백 단위로 분리
        row = tsv.Split(' ');
    }

    public string RoadData(string key)
    {
        Debug.Log(key);
        //불러온 스프리드 시트의 인덱스 번호 중 key와 같은 단어를 찾는다
        for (int i = 0; i < row.Length; i++)
        {
            string[] column = row[i].Split(' ');  //공백 단위로 문자열 나눔 column에는 공백 개수+1개의 문자열이 저장

            if (column[0] == key)
            {  //key와 같은 한글 단어를 찾는 다면 그와 쌍으로 존재하는 영단어 반환
                column[0] = column[1].Replace("\r", "");
                return column[0];
            }
        }
        //없으면 공백 반환
        Debug.Log("..");
        return "";
    }
}
