/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package com.xiaomi.chatbot.services.nsasearch.model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.IntBuffer;
import java.util.*;
import java.util.Comparator;
import java.math.BigDecimal;

import com.xiaomi.chatbot.common.util.LogUtil;
import com.xiaomi.chatbot.services.common.wrapper.ics.Constant;

import org.apache.commons.lang3.StringUtils;
import lombok.extern.slf4j.Slf4j;

import org.tensorflow.Session;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.WordDictionary;

import com.xiaomi.chatbot.services.common.wrapper.ics.SearchQuery;
//import com.xiaomi.chatbot.services.cssearch.wrapper.faq.FaqResult;
import com.xiaomi.chatbot.services.common.wrapper.ics.essearch.FaqResult;

@Slf4j
public class NsaRanking {

    private SavedModelBundle model_bundle = null;
    private Session model_session = null;
    private JiebaSegmenter segmenter = new JiebaSegmenter();
    private Map word_index = null;
    private static int MAX_LEN = 20;

    private static String nsaRankingModelDir       = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-phone";
    private static String nsaRankingIHModelDir     = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-ih";
    private static String nsaRankingTVModelDir     = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-tv";
    private static String nsaRankingACModelDir     = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-ac";
    private static String nsaRankingRouterModelDir = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-router";
    private static String nsaRankingNbModelDir     = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-nb";
    private static String nsaRankingPhoneCardModelDir     = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-phonecard";
    private static String nsaRankingNewNetBankModelDir     = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-newnet";

    private static NsaRanking INSTANCE_PHONE = null;
    private static NsaRanking INSTANCE_IH = null;
    private static NsaRanking INSTANCE_TV = null;
    private static NsaRanking INSTANCE_AC = null;
    private static NsaRanking INSTANCE_ROUTER = null;
    private static NsaRanking INSTANCE_NB = null;
    private static NsaRanking INSTANCE_PHONECARD = null;
    private static NsaRanking INSTANCE_NEWNET = null;

    static {
        INSTANCE_PHONE = new NsaRanking(nsaRankingModelDir);
        INSTANCE_IH = new NsaRanking(nsaRankingIHModelDir);
        INSTANCE_TV = new NsaRanking(nsaRankingTVModelDir);
        INSTANCE_AC = new NsaRanking(nsaRankingACModelDir);
        INSTANCE_ROUTER = new NsaRanking(nsaRankingRouterModelDir);
        INSTANCE_NB = new NsaRanking(nsaRankingNbModelDir);
        INSTANCE_PHONECARD = new NsaRanking(nsaRankingPhoneCardModelDir);
        INSTANCE_NEWNET = new NsaRanking(nsaRankingNewNetBankModelDir);
    }

    public static NsaRanking getPhoneInstance() {
        return INSTANCE_PHONE;
    }
    public static NsaRanking getIhInstance() {
        return INSTANCE_IH;
    }
    public static NsaRanking getTvInstance() {
        return INSTANCE_TV;
    }
    public static NsaRanking getAcInstance() {
        return INSTANCE_AC;
    }
    public static NsaRanking getRouterInstance() {
        return INSTANCE_ROUTER;
    }
    public static NsaRanking getNbInstance() {
        return INSTANCE_NB;
    }
    public static NsaRanking getPhoneCardInstance() {
        return INSTANCE_PHONECARD;
    }
    public static NsaRanking getNewNetInstance() {
        return INSTANCE_NEWNET;
    }

    public NsaRanking(String modelPath) {
        model_bundle = SavedModelBundle.load(modelPath, "serve");
        model_session = model_bundle.session();
        String entity_file = modelPath + "/userdict.txt";
        Path entity_path = Paths.get(entity_file);
        WordDictionary.getInstance().loadUserDict(entity_path);
        String word_file = modelPath + "/word.tsv";
        word_index = loadWordMap(word_file);
        session_init();
        log.info("load nsa model: {} finished", modelPath);
    }

    public void session_init() {

        long[] query_shape = new long[]{1, MAX_LEN};
        long[] query_len_shape = new long[] {1};

        Tensor batch_size = Tensor.create(1, Integer.class);
        Tensor dropout_keep_prob = Tensor.create(1.0f, Float.class);

        IntBuffer query_ids = IntBuffer.allocate(1 * MAX_LEN);
        IntBuffer candidate_ids = IntBuffer.allocate(1 * MAX_LEN);

        IntBuffer query_len_buf = IntBuffer.allocate(1);
        IntBuffer candidate_len_buf = IntBuffer.allocate(1);

        for(int i=0; i < 1; i++) {
            String candidate = "我想查询已购买产品发货/到货进度"; 
            int word_len = word2id(candidate, candidate_ids);
            candidate_len_buf.put(20);

            String query = "什么时候能到货";
            word_len = word2id(query, query_ids);
            query_len_buf.put(word_len);
        }

        query_ids.flip();
        Tensor<Integer> query_input = Tensor.create(query_shape, query_ids);

        candidate_ids.flip();
        Tensor<Integer> candidate_input = Tensor.create(query_shape, candidate_ids);

        query_len_buf.flip();
        Tensor<Integer> query_len = Tensor.create(query_len_shape, query_len_buf);

        candidate_len_buf.flip();
        Tensor<Integer> candidate_len = Tensor.create(query_len_shape, candidate_len_buf);

        long startTime = System.currentTimeMillis();
        Tensor<Float> result =
              model_session
              .runner()
              .feed("encoder_inputs", query_input)
              .feed("encoder_inputs_actual_lengths", query_len)
              .feed("decoder_outputs", candidate_input)
              .feed("decoder_outputs_actual_lengths", candidate_len)
              .feed("batch_size", batch_size)
              .feed("dropout_keep_prob", dropout_keep_prob)
              .fetch("train_predict_prob")
              .run()
              .get(0).expect(Float.class);

        long endTime   = System.currentTimeMillis();
        long TotalTime = endTime - startTime;
        if (log.isDebugEnabled()) {
            log.debug("session_init, cost {} ms", (int) (TotalTime));
        }
        query_input.close();
        candidate_input.close();

        query_len.close();
        candidate_len.close();

        batch_size.close();
        dropout_keep_prob.close();
        result.close();
    }

    public boolean initialize(String modelPath) {
        try {
            model_bundle = SavedModelBundle.load(modelPath, "serve");
            String entity_file = modelPath + "/userdict.txt";
            Path entity_path = Paths.get(entity_file);
            WordDictionary.getInstance().loadUserDict(entity_path);
            String word_file = modelPath + "/word.tsv";
            word_index = loadWordMap(word_file);
        }
        catch (Exception e) {
            log.error("NsaRaning initialize failed: {}", e);
            return false;
        }
        return true;
    }

    public Map loadWordMap(String fileName) {
        Map word_index = new HashMap();
        File file = new File(fileName);
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(file));
            String tempString = null;
            int line = 1;
            while ((tempString = reader.readLine()) != null) {
                String[] buff = tempString.split("\t");
                int index = Integer.parseInt(buff[0]);
                String word = buff[1];
                word_index.put(word, index);
            }
        }
        catch (Exception e) {
            log.error("nsa config error: {}", e);
            return word_index;
        }
        return word_index;
    }

    public int mapValue2int(String word) {
        return Integer.parseInt(word_index.get(word).toString());
    }

    public int word2id(String candidate, IntBuffer candidate_ids) {
        int word_count = 0;
        int word_len = 0;
        for (String word : segmenter.sentenceProcess(candidate)) {
            if (word_index.containsKey(word)) {
                if (word_count == MAX_LEN-1) {
                    //candidate_ids.put(mapValue2int("</s>"));
                    //word_len = MAX_LEN-1;
                    break;
                } else {
                    candidate_ids.put(mapValue2int(word));
                    word_count += 1;
                }
            }
        }
        if(word_count == MAX_LEN-1) {
            //System.out.println(word);
            candidate_ids.put(mapValue2int("</s>"));
            word_len = MAX_LEN-1; 
        }
        word_len = word_count;
        if(word_count < MAX_LEN-1) {
            candidate_ids.put(mapValue2int("</s>"));
            word_count += 1;
            while(word_count < MAX_LEN) {
                candidate_ids.put(mapValue2int("<pad>"));
                word_count += 1;
            }   
        }
        return word_len;
    }

    public List<FaqResult> getNsaRankingResult(SearchQuery searchQuery) {
        long begin = System.currentTimeMillis();

        List<FaqResult> esResults = searchQuery.getEsResults();
        if (esResults == null) {
            return null;
        }
        String requestId = searchQuery.getRequestId();
        String bucket = searchQuery.getBucket();
        String skill = searchQuery.getSkill();
        //Map<String, GroupInfo> groupInfo = searchQuery.getGroupInfo();
        String group = "";
        for (String key: searchQuery.getGroupInfo().keySet()) {
            group = key;
        }
        if (!group.equals("小米网") && !group.equals("新网银行")) {
            return esResults;
        }
        //boolean isNewnetGroup = searchQuery.getGroupInfo().containsKey("新网银行");
        int candidate_num = esResults.size();

        //String query = searchQuery.getQuery();
        String query = searchQuery.getGroupInfo().get(group).getRevisedQuery();
        //Map<String, List<String>> parentEntities = searchQuery.getGroupInfo().get("小米网").getParentEntities();

        if (query == null || query.isEmpty()) {
            query = searchQuery.getQuery();
        }

        String query_lower = query.toLowerCase();

        long[] query_shape = new long[]{candidate_num, 20};
        long[] query_len_shape = new long[]{candidate_num};

        Tensor batch_size = Tensor.create(candidate_num, Integer.class);
        Tensor dropout_keep_prob = Tensor.create(1.0f, Float.class);

        IntBuffer query_ids = IntBuffer.allocate(candidate_num * 20);
        IntBuffer candidate_ids = IntBuffer.allocate(candidate_num * 20);
        IntBuffer query_len_buf = IntBuffer.allocate(candidate_num);
        IntBuffer candidate_len_buf = IntBuffer.allocate(candidate_num);

        int query_wordlen = 0;

        for (FaqResult es_info : esResults) {
            //String match_query = query_lower;
            String candidate = es_info.getRawSimQuestion();
            String candidate_level = es_info.getLevel(); 
            int word_len = 0;
            //String candidate = es_info.getSimQuestion();
            if (group.equals("新网银行") && query_lower.indexOf("好人贷") == -1 && candidate.indexOf("好人贷") != -1) {
                word_len = word2id(candidate.replace("好人贷", "<xw_loan>"), candidate_ids);
            } else {
                word_len = word2id(candidate, candidate_ids);
            }
            //candidate_len_buf.put(word_len);
            candidate_len_buf.put(20);

            query_wordlen = word2id(query_lower, query_ids);
            //query_wordlen = word2id(match_query, query_ids);
            query_len_buf.put(query_wordlen);
        }
        if (query_wordlen < 1) {
            return esResults;
        }

        query_ids.flip();
        Tensor<Integer> query_input = Tensor.create(query_shape, query_ids);

        candidate_ids.flip();
        Tensor<Integer> candidate_input = Tensor.create(query_shape, candidate_ids);

        query_len_buf.flip();
        Tensor<Integer> query_len = Tensor.create(query_len_shape, query_len_buf);

        candidate_len_buf.flip();
        Tensor<Integer> candidate_len = Tensor.create(query_len_shape, candidate_len_buf);

        Tensor<Float> result = model_session
                .runner()
                .feed("encoder_inputs", query_input)
                .feed("encoder_inputs_actual_lengths", query_len)
                .feed("decoder_outputs", candidate_input)
                .feed("decoder_outputs_actual_lengths", candidate_len)
                .feed("batch_size", batch_size)
                .feed("dropout_keep_prob", dropout_keep_prob)
                .fetch("train_predict_prob")
                .run()
                .get(0)
                .expect(Float.class);

        query_input.close();
        candidate_input.close();
        query_len.close();
        candidate_len.close();
        batch_size.close();
        dropout_keep_prob.close();

        final long[] rshape = result.shape();

        int nlabels = (int)rshape[1];
        float[][] candidate_prob = result.copyTo(new float[candidate_num][nlabels]);
        result.close();
        if (log.isDebugEnabled()) {
            log.debug("requestId: {}, bucket: {}, msg: debug_nsa_ranking\tuser_query:{}", requestId, bucket, query_lower);
        }
        float max_score = 0;
        String final_stdQ = "";
        for (int i = 0; i < candidate_num; i++) {
            FaqResult candidate_esinfo = esResults.get(i);
            String candidate = candidate_esinfo.getRawSimQuestion();
            String stdQ = candidate_esinfo.getOrigQuestion();
            String level = candidate_esinfo.getLevel();
            float predict_score = candidate_prob[i][1];
            predict_score = predict_score / 1.06f;

            if (predict_score - 0.6 < 0.000001) {
                predict_score = predict_score / 2.4f;
            }
            if (StringUtils.isBlank(level) || level.startsWith(Constant.MIWEB_CHAT_LEVEL_PREFIX)) {
                predict_score = 0.0f;
            }

            if (log.isDebugEnabled()) {
                log.debug("requestId: {}, bucket: {}, msg: debug_nsa_ranking\tcandidate:{}\tstdQ:{}\t{}", requestId, bucket, candidate, stdQ, predict_score);
            }

            BigDecimal b = new BigDecimal(String.valueOf(predict_score));  
            double predict_score_double = b.doubleValue(); 
            candidate_esinfo.setScore(predict_score_double);
            esResults.set(i, candidate_esinfo);

            if (predict_score - max_score > 0.000001) {
                final_stdQ = candidate_esinfo.getOrigQuestion();
                max_score = predict_score;
            }
        }
        if (log.isDebugEnabled()) {
            log.debug("requestId:{}, bucket: {}, msg: debug_nsa_ranking\tquery:{}\tfinal_stdQ:{}\t{}", requestId, bucket, query, final_stdQ, max_score);
        }
        LogUtil.timeCost(requestId, "inside nsa-search", System.currentTimeMillis()-begin);
        return esResults;
    }

    public static void main(String[] args) {
        System.out.println("debug into main");
        String modelDir = args[0];
        /*
        NsaRanking nsa_ranking = NsaRanking.getInstance(); 
        //if(!nsa_ranking.initialize(modelDir)) {
        //    System.out.println("");
        //}
        System.out.println("init success");
        String query = "红米note4x什么时间会有货";
        String candidate_query = "红米note4x什么时间有货";
        for(int i=0; i < 100; i++) {
            List<FaqResult> nsaResult = nsa_ranking.getNsaRankingResult(esResults, searchQuery);
        }
        */
    }
}

class StrLenComparator implements Comparator<String>{  
  
    @Override  
    public int compare(String o1, String o2) {  
        if(o1.length()>o2.length())  
            return -1;  
        if(o1.length()<o2.length())  
            return 1;  
        return o1.compareTo(o2);  
    }  
      
} 
