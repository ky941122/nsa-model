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

import java.io.IOException;
import java.io.PrintStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.IntBuffer;
import java.util.*;
import java.util.Collections;  
import java.util.Comparator; 
import java.math.BigDecimal;

import com.xiaomi.chatbot.services.common.wrapper.ics.Constant;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.UInt8;
import com.huaban.analysis.jieba.JiebaSegmenter;
import com.huaban.analysis.jieba.WordDictionary;

import com.xiaomi.chatbot.services.common.wrapper.ics.SearchQuery;
import com.xiaomi.chatbot.services.common.wrapper.ics.SearchResult;
//import com.xiaomi.chatbot.services.cssearch.wrapper.faq.FaqResult;
import com.xiaomi.chatbot.services.common.wrapper.ics.essearch.FaqResult;

@Slf4j
public class NsaRankingV2 {

    private SavedModelBundle model_bundle = null;
    private Session model_session = null;
    private JiebaSegmenter segmenter = new JiebaSegmenter();
    private Map word_index = null;
    private Map level_index = null;
    private static int MAX_LEN = 20;
    private static int MAX_LEVEL_LEN = 6;

    private static String nsaRankingMiBankModelDir     = "/home/work/chatbot/files/nsa_model/nsa-ranking-model-mibank";

    private static NsaRankingV2 INSTANCE_MIBANK = null;

    static {
        INSTANCE_MIBANK = new NsaRankingV2(nsaRankingMiBankModelDir);
    }

    public static NsaRankingV2 getMiBankInstance() {
        return INSTANCE_MIBANK;
    }

    public NsaRankingV2(String modelPath) {
        model_bundle = SavedModelBundle.load(modelPath, "serve");
        model_session = model_bundle.session();
        String entity_file = modelPath + "/userdict.txt";
        Path entity_path = Paths.get(entity_file);
        WordDictionary.getInstance().loadUserDict(entity_path);
        String word_file = modelPath + "/word.tsv";
        String level_file = modelPath + "/level.tsv";
        word_index = loadWordMap(word_file);
        level_index = loadWordMap(level_file);
        session_init();
        log.info("load nsa model: {} finished", modelPath);
    }

    public void session_init() {

        long[] query_shape = new long[]{1, MAX_LEN};
        long[] levels_shape = new long[]{1, MAX_LEVEL_LEN};
        long[] query_len_shape = new long[] {1};

        Tensor batch_size = Tensor.create(1, Integer.class);
        Tensor dropout_keep_prob = Tensor.create(1.0f, Float.class);

        IntBuffer query_ids = IntBuffer.allocate(1 * MAX_LEN);
        IntBuffer candidate_ids = IntBuffer.allocate(1 * MAX_LEN);

        IntBuffer query_len_buf = IntBuffer.allocate(1);
        IntBuffer candidate_len_buf = IntBuffer.allocate(1);

        IntBuffer levels_ids = IntBuffer.allocate(1 * MAX_LEVEL_LEN);

        for(int i=0; i < 1; i++) {
            String candidate = "我想查询已购买产品发货/到货进度"; 
            int word_len = word2id(candidate, candidate_ids);
            candidate_len_buf.put(20);

            String query = "什么时候能到货";
            word_len = word2id(query, query_ids);
            query_len_buf.put(word_len);

            String levels = "小米网:TOP问:首页TOP:NULL:NULL:NULL";
            levels2id(levels, levels_ids);
        }

        query_ids.flip();
        Tensor<Integer> query_input = Tensor.create(query_shape, query_ids);

        candidate_ids.flip();
        Tensor<Integer> candidate_input = Tensor.create(query_shape, candidate_ids);

        query_len_buf.flip();
        Tensor<Integer> query_len = Tensor.create(query_len_shape, query_len_buf);

        candidate_len_buf.flip();
        Tensor<Integer> candidate_len = Tensor.create(query_len_shape, candidate_len_buf);

        levels_ids.flip();
        Tensor<Integer> levels_input = Tensor.create(levels_shape, levels_ids);

        long startTime = System.currentTimeMillis();
        Tensor<Float> result =
              model_session
              .runner()
              .feed("encoder_inputs", query_input)
              .feed("encoder_inputs_actual_lengths", query_len)
              .feed("decoder_outputs", candidate_input)
              .feed("decoder_outputs_actual_lengths", candidate_len)
              .feed("levels", levels_input)
              .feed("batch_size", batch_size)
              .feed("dropout_keep_prob", dropout_keep_prob)
              .fetch("train_predict_prob")
              .run()
              .get(0).expect(Float.class);

        long endTime   = System.currentTimeMillis();
        long TotalTime = endTime - startTime;
        log.debug("session_init, cost {} ms", (int) (TotalTime));

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
            log.error("NsaRaningV2 initialize failed: {}", e);
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
            log.info("nsa config error: {}", e);
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

    public void levels2id(String levels, IntBuffer levels_ids) {
        int level_count = 0;
        for (String level : levels.split(":")) {
            if (level_index.containsKey(level)) {
                levels_ids.put(Integer.parseInt(level_index.get(level).toString()));
                level_count += 1;
            }
        }
        while(level_count < MAX_LEVEL_LEN) {
            levels_ids.put(Integer.parseInt(level_index.get("NULL").toString()));
            level_count += 1;
        }
    }
    public List<FaqResult> getRankingResult(SearchQuery searchQuery) {
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
        if (!group.equals("小米网") && !group.equals("新网银行") && !group.equals("金融服务")) {
            return esResults;
        }

        int candidate_num = esResults.size();

        //String query = searchQuery.getQuery();
        String query = searchQuery.getGroupInfo().get(group).getRevisedQuery();
        //Map<String, List<String>> parentEntities = searchQuery.getGroupInfo().get("小米网").getParentEntities();

        if (query == null || query.isEmpty()) {
            query = searchQuery.getQuery();
        }

        String query_lower = query.toLowerCase();

        long[] query_shape = new long[]{candidate_num, MAX_LEN};
        long[] levels_shape = new long[]{candidate_num, MAX_LEVEL_LEN};
        long[] query_len_shape = new long[]{candidate_num};

        Tensor batch_size = Tensor.create(candidate_num, Integer.class);
        Tensor dropout_keep_prob = Tensor.create(1.0f, Float.class);

        IntBuffer query_ids = IntBuffer.allocate(candidate_num * MAX_LEN);
        IntBuffer candidate_ids = IntBuffer.allocate(candidate_num * MAX_LEN);
        IntBuffer query_len_buf = IntBuffer.allocate(candidate_num);
        IntBuffer candidate_len_buf = IntBuffer.allocate(candidate_num);
        IntBuffer levels_ids = IntBuffer.allocate(candidate_num * MAX_LEVEL_LEN);

        int query_wordlen = 0;

        for (FaqResult es_info : esResults) {
            //String match_query = query_lower;
            String candidate = es_info.getRawSimQuestion();
            String candidate_level = es_info.getLevel();
            String levels = es_info.getLevel();
            int word_len = 0;
            //String candidate = es_info.getSimQuestion();
            if (group.equals("新网银行") && query_lower.indexOf("好人贷") == -1 && candidate.indexOf("好人贷") != -1) {
                word_len = word2id(candidate.replace("好人贷", "<xw_loan>"), candidate_ids);
            } else {
                word_len = word2id(candidate, candidate_ids);
            }
            //candidate_len_buf.put(word_len);
            candidate_len_buf.put(MAX_LEN);

            query_wordlen = word2id(query_lower, query_ids);
            //query_wordlen = word2id(match_query, query_ids);
            query_len_buf.put(query_wordlen);

            levels2id(levels, levels_ids);
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

        levels_ids.flip();
        Tensor<Integer> levels_input = Tensor.create(levels_shape, levels_ids);

        Tensor<Float> result = model_session
                .runner()
                .feed("encoder_inputs", query_input)
                .feed("encoder_inputs_actual_lengths", query_len)
                .feed("decoder_outputs", candidate_input)
                .feed("decoder_outputs_actual_lengths", candidate_len)
                .feed("levels",levels_input)
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
        log.debug("requestId: {}, bucket: {}, msg: debug_nsa_ranking\tuser_query:{}", requestId, bucket, query_lower);
        //LOGGER.info(String.format("debug_nsa_ranking\tquery_segment:%s", segmenter.sentenceProcess(query)));
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
            if (StringUtils.isBlank(level) || level.indexOf("chat") != -1 || level.indexOf("CHAT") != -1) {
                predict_score = 0.0f;
            }
            log.debug("requestId: {}, bucket: {}, msg: debug_nsa_ranking\tcandidate:{}\tstdQ:{}\t{}", requestId, bucket, candidate, stdQ, predict_score);

            BigDecimal b = new BigDecimal(String.valueOf(predict_score));  
            double predict_score_double = b.doubleValue(); 
            candidate_esinfo.setScore(predict_score_double);
            esResults.set(i, candidate_esinfo);

            if (predict_score - max_score > 0.000001) {
                final_stdQ = candidate_esinfo.getOrigQuestion();
                max_score = predict_score;
            }
        }
        log.debug("requestId:{}, bucket: {}, msg: debug_nsa_ranking\tquery:{}\tfinal_stdQ:{}\t{}", requestId, bucket, query, final_stdQ, max_score);
        log.debug("requestId:{}, bucket: {}, msg: success return nsa-search result, procedure=inside nsa-search, time={}, query: {}", requestId, bucket, (int) (System.currentTimeMillis() - begin), query);
        return esResults;
    }

    public static void main(String[] args) {
        System.out.println("debug into main");
        String modelDir = args[0];
        /*
        NsaRankingV2 nsa_ranking = NsaRankingV2.getInstance(); 
        //if(!nsa_ranking.initialize(modelDir)) {
        //    System.out.println("");
        //}
        System.out.println("init success");
        String query = "红米note4x什么时间会有货";
        String candidate_query = "红米note4x什么时间有货";
        for(int i=0; i < 100; i++) {
            List<FaqResult> nsaResult = nsa_ranking.getNsaRankingV2Result(esResults, searchQuery);
        }
        */
    }
}

