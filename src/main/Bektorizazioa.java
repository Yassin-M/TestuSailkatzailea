package main;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.NullStemmer;
import weka.core.stemmers.Stemmer;
import weka.core.stopwords.Null;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

public class Bektorizazioa {
    private static StringToWordVector stwv;
    private static FixedDictionaryStringToWordVector fdstwv;
    private static AttributeSelection as;

    public static void main(String[] args) throws Exception{
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("data/arff/tweetSentiment.train.arff");
        Instances data = source.getDataSet();
        konfigurazioEgokienaAukeratu(data);
        //proba
        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource("data/arff/tweetSentiment.dev.arff");
        Instances test = source1.getDataSet();
        stwv.setInputFormat(data);
        Instances trainBek = Filter.useFilter(data, stwv);
        as.setInputFormat(trainBek);
        setFdstwv();
        Instances trainFinal = Filter.useFilter(trainBek, as);
        test.setClassIndex(test.attribute("Sentiment").index());
        fdstwv.setInputFormat(test);
        Instances testBek = Filter.useFilter(test, fdstwv);
        as.setInputFormat(trainBek);
        Instances testFinal = Filter.useFilter(testBek, as);
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(trainFinal);
        Evaluation eval = new Evaluation(trainFinal);
        eval.evaluateModel(nb, testFinal);
        System.out.println("Resultados finales sobre el set de TEST_BLIND:");
        System.out.println("Accuracy: " + eval.pctCorrect() + "%");

    }

    public static void konfigurazioEgokienaAukeratu(Instances train) throws Exception{
        if(train.classIndex()==-1) train.setClassIndex(train.attribute("Sentiment").index());
        // probatu nahi ditugun aukerak
        boolean bestStem = false;
        int bestN = 1;
        double bestAccuracy = -1.0;
        int bestMota = -1;
        int bestTok = -1;
        int i = 1;
        for(int mota = 0; mota<3; mota++){
            for(boolean stemmer: new boolean[]{false, true}){// stemmer erabili ala ez
                for(int tokenizer=0; tokenizer<=1; tokenizer++){
                    StringToWordVector unekoFiltroa = new StringToWordVector();
                    unekoFiltroa.setWordsToKeep(1000);
                    unekoFiltroa.setLowerCaseTokens(true);
                    unekoFiltroa.setAttributeNamePrefix("W_");
                    unekoFiltroa.setDictionaryFileToSaveTo(new File("./dictionary.txt"));
                    //Bektorizazio mota egokiena aukeratu
                    filtroaPrestatu(unekoFiltroa, mota);
                    //Stemmer-a aukeratu
                    stemmerEzarri(unekoFiltroa, stemmer);
                    //Tokenizer aukeratu
                    tokenizerEzarri(unekoFiltroa, tokenizer);

                    unekoFiltroa.setInputFormat(train);
                    Instances bektorizatuak = Filter.useFilter(train, unekoFiltroa);

                    AttributeSelection unekoAS = new AttributeSelection();
                    InfoGainAttributeEval eval = new InfoGainAttributeEval();
                    Ranker ranker = new Ranker();
                    ranker.setNumToSelect(500);

                    unekoAS.setEvaluator(eval);
                    unekoAS.setSearch(ranker);
                    unekoAS.setInputFormat(bektorizatuak);

                    Instances finala = Filter.useFilter(bektorizatuak, unekoAS);

                    Evaluation ev = new Evaluation(finala);
                    ev.crossValidateModel(new NaiveBayes(), finala, 10, new Random(i));

                    double acc = ev.pctCorrect();
                    System.out.println("Mota: " + getMotaIzena(mota) + " | Stem: " + getStemmerIzena(stemmer) + " | Tok: " + getTokenizerIzena(tokenizer) + " -> Acc: " + acc);

                    if (acc > bestAccuracy) {
                        bestAccuracy = acc;
                        bestMota = mota;
                        bestStem = stemmer;
                        bestTok = tokenizer;
                        setStwv(unekoFiltroa);
                        as = unekoAS;
                    }
                    i++;
                }
            }
        }
        System.out.println("Hoberena --> Mota: " + getMotaIzena(bestMota) + " | Stem: " + getStemmerIzena(bestStem) + " | Tok: " + getTokenizerIzena(bestTok) + " -> Acc: " + bestAccuracy);
        //Informazio guztia gorde
        SerializationHelper.write("dataFinala/filter/bestAttributeSelection.filter", as);
        String[] config = stwv.getOptions();
        String configTxt = Utils.joinOptions(config);
        Files.write(Paths.get("dataFinala/txt/bektorizazioHoberena.txt"), configTxt.getBytes());
        stwv.getDictionaryFileToSaveTo().renameTo(new File("dataFinala/txt/bestDictionary.txt"));

    }

    public static void setFdstwv(){
        StringToWordVector stringToWordVector = getStwv();
        fdstwv = new FixedDictionaryStringToWordVector();
        fdstwv.setTFTransform(stringToWordVector.getTFTransform());
        fdstwv.setIDFTransform(stringToWordVector.getIDFTransform());
        fdstwv.setOutputWordCounts(stringToWordVector.getOutputWordCounts());
        fdstwv.setDictionaryFile(stringToWordVector.getDictionaryFileToSaveTo());
        fdstwv.setTokenizer(stringToWordVector.getTokenizer());
        fdstwv.setStemmer(stringToWordVector.getStemmer());

    }

    public static StringToWordVector getStwv(){
        return stwv;
    }

    public static void setStwv(StringToWordVector pStwv){
        stwv = pStwv;
    }

    private static void filtroaPrestatu(StringToWordVector f, int mota){
        if(mota==0){
            //Bektorizazio bitarra
            f.setOutputWordCounts(false);
            f.setTFTransform(false);
            f.setIDFTransform(false);
        }else if(mota == 1){
            //TF
            f.setOutputWordCounts(true);
            f.setTFTransform(true);
            f.setIDFTransform(false);
        }else if(mota == 2){
            //TF-IDF
            f.setOutputWordCounts(true);
            f.setTFTransform(true);
            f.setIDFTransform(true);
        }
    }

    private static void stemmerEzarri(StringToWordVector f, boolean stem){
        if(stem){
            f.setStemmer(new IteratedLovinsStemmer());
        }else{
            f.setStemmer(new NullStemmer());
        }
    }

    private static void tokenizerEzarri(StringToWordVector f, int tokenizer){
        if(tokenizer==0){
            f.setTokenizer(new AlphabeticTokenizer());
        }else{
            NGramTokenizer ngram = new NGramTokenizer();
            ngram.setNGramMinSize(1);
            ngram.setNGramMaxSize(2);
            f.setTokenizer(ngram);
        }
    }

    private static String getMotaIzena(int mota) {
        return switch (mota) {
            case 0 -> "Bitarra";
            case 1 -> "TF";
            case 2 -> "TF-IDF";
            default -> "Ezezaguna";
        };
    }

    private static String getStemmerIzena(boolean stem) {
        return stem ? "IteratedLovins" : "None";
    }

    private static String getTokenizerIzena(int tok) {
        return (tok == 0) ? "Alphabetic" : "NGram(1-2)";
    }
}
