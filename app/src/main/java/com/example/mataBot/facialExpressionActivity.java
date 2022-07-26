package com.example.mataBot;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

public class facialExpressionActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="MainActivity";

    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    // call java class
    private facialExpressionClass facialExpressionClass;
    private ImageView flip_camera;
    private int mCameraId=0;

    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public facialExpressionActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(facialExpressionActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(facialExpressionActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_facial_expression);

        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        flip_camera=findViewById(R.id.flip_camera);
        flip_camera.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                if(motionEvent.getAction()==MotionEvent.ACTION_DOWN){
                    flip_camera.setColorFilter(Color.DKGRAY);
                    return true;
                }
                if(motionEvent.getAction()==MotionEvent.ACTION_UP){
                    flip_camera.setColorFilter(Color.WHITE);
                    swapCamera();
                    return true;
                }
                return false;
            }
        });

        // this will load cascade classifier and model
        // this only happen one time when you start CameraActivity
        try{
            // input size of model is 48
            int inputSize=48;
            facialExpressionClass=new facialExpressionClass(getAssets(),facialExpressionActivity.this,
                    "facial_expression_model.tflite",inputSize);
        }
        catch (IOException e){
            e.printStackTrace();
        }


    }

    private void swapCamera() {
        mCameraId=mCameraId^1;
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(mCameraId);
        mOpenCvCameraView.enableView();
        mOpenCvCameraView.enableFpsMeter();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            //if load success
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if not loaded
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);
    }
    public void onCameraViewStopped(){
        mRgba.release();
    }
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        mRgba=inputFrame.rgba();
        mGray=inputFrame.gray();

        if(mCameraId==1){
            Core.flip(mRgba,mRgba, 1);
            Core.flip(mGray,mGray, -1);

        }

        //output                                         input
        mRgba=facialExpressionClass.recognizeImage(mRgba);
        return mRgba;

    }

}