package com.kku.jcompareml;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;

/**
 * Created by off on 1/25/15.
 */
public class CompareML {

    public static void main(String [] args){
        Classifier cls = new NaiveBayes();
        System.out.println(cls);
        try {
            cls.buildClassifier(null);
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("hello world");
    }
}
