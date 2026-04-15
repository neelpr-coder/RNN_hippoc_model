using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class TwoDimMovement : MonoBehaviour
{   
    private int fileCounter = 0;
    void CamCapture(float i, float j, int k, int fileCounter)
    {
        Camera Cam = this.GetComponent<Camera>();

        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = Cam.targetTexture;
 
        Cam.Render();

        Texture2D Image = new Texture2D(Cam.targetTexture.width, Cam.targetTexture.height);
        Image.ReadPixels(new Rect(0, 0, Cam.targetTexture.width, Cam.targetTexture.height), 0, 0);
        Image.Apply();
        RenderTexture.active = currentRT;
 
        var Bytes = Image.EncodeToPNG();
        Destroy(Image);
        File.WriteAllBytes($"/Users/neelprabhakar/Desktop/unity/2d/{fileCounter:D6}.png", Bytes);
    }

    // Start is called before the first frame update
    void Start()
    {
        Directory.CreateDirectory("/Users/neelprabhakar/Desktop/unity/RNN_images");

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    //FixedUpdate
    void FixedUpdate()
    {   
        //int fileCounter = 0;
        
        float x_min = -1f;
        float x_max = 1f;
        float z_min = -1f;
        float z_max = 1f;
        float multiplier = 8;   //increase the sampling rate by 10 so more intermediate positions are sampled
       
        float n_rotations = 4f;
        float total_degrees = 360f;
        float rotate_degree = total_degrees / n_rotations;

        //place robot to a new position based on 2d array from -17 to 17
        for (float i = x_min*multiplier; i <= x_max*multiplier; i++)
        {
            for (float j = z_min*multiplier; j <= z_max*multiplier; j++)
            {   
                float x_coord = i/multiplier;
                float z_coord = j/multiplier;
                transform.localPosition = new Vector3(x_coord, 0.25f, z_coord);

                //rotate camera k times by 360/k degrees
                for (int k = 0; k < n_rotations; k++)
                {   
                    transform.localRotation = Quaternion.Euler(0f, k * rotate_degree, 0f);
                    CamCapture(x_coord, z_coord, k, fileCounter);
                    //transform.Rotate(0f, rotate_degree, 0f);
                    Debug.Log($"x: {x_coord}, z: {z_coord}, k: {k}");
                    fileCounter++;
                }
            }
        }
        //exit play mode
        UnityEditor.EditorApplication.isPlaying = false;
        enabled = false;
    }
}   
