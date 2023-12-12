using UnityEngine;

namespace YOLOv3MLNet.DataStructures
{
    public class YoloV3BitmapData
    {
        //[ColumnName("bitmap")]
        //[ImageType(416, 416)]
        public Texture Image { get; set; }

        //[ColumnName("width")]
        public float ImageWidth => Image.width;

        //[ColumnName("height")]
        public float ImageHeight => Image.height;
    }
}