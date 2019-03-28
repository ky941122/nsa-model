package com.xiaomi.chatbot.services.nsasearch.controller;

import com.xiaomi.chatbot.services.common.wrapper.ics.Constant;
import com.xiaomi.chatbot.services.nsasearch.model.NsaRanking;
import com.xiaomi.chatbot.services.nsasearch.model.NsaRankingV2;
import com.xiaomi.chatbot.services.common.wrapper.ics.essearch.FaqResult;
import com.xiaomi.chatbot.common.Singleton;
import com.xiaomi.chatbot.services.common.wrapper.ics.SearchQuery;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.lang3.StringUtils;

import java.util.*;
import java.util.concurrent.*;

/**
 * Created by guoxiang1 on 18-07-16.
 */

@Slf4j
public class NsaSearchController {

    //private CnnRanking cnnRecallAll = CnnRanking.getAllInstance();
    private NsaRanking nsaRankingPhone = NsaRanking.getPhoneInstance();
    private NsaRanking nsaRankingIh = NsaRanking.getIhInstance();
    private NsaRanking nsaRankingTv = NsaRanking.getTvInstance();
    private NsaRanking nsaRankingAc = NsaRanking.getAcInstance();
    private NsaRanking nsaRankingRouter = NsaRanking.getRouterInstance();
    private NsaRanking nsaRankingNb = NsaRanking.getNbInstance();
    private NsaRanking nsaRankingPhoneCard = NsaRanking.getPhoneCardInstance();
    private NsaRanking nsaRankingNewNet = NsaRanking.getNewNetInstance();
    //private NsaRankingV2 nsaRankingNewNet = NsaRankingV2.getNewNetInstance();
    private NsaRankingV2 nsaRankingMiBank = NsaRankingV2.getMiBankInstance();
    //private NsaRankingV2 nsaRankingRouter = NsaRankingV2.getRouterInstance();

    public String getResult(SearchQuery searchQuery) {
        if (searchQuery == null) {
            log.warn("Error params: SearchQuery is null");
            return null;
        }

        List<FaqResult> nsaResults = new ArrayList<>();

        if (searchQuery.getGroupInfo().keySet().contains("小米网") || searchQuery.getGroupInfo().keySet().contains("小米网其他") || searchQuery.getGroupInfo().keySet().contains("小米网智能硬件测试group")) {
            // 开启线程请求nsa-search
            String skill = searchQuery.getSkill();
            if (StringUtils.isBlank(skill)) {
                if (log.isDebugEnabled()) {
                    log.debug("Skill empty, cancel ask nsa");
                }
            } else if (skill.equals(Constant.MIWEB_SKILL_HARDWARE)) {
                nsaResults = nsaRankingIh.getNsaRankingResult(searchQuery);
            } else if (skill.equals(Constant.MIWEB_SKILL_AC)) {
                nsaResults = nsaRankingAc.getNsaRankingResult(searchQuery);
            } else if (skill.equals(Constant.MIWEB_SKILL_TV)) {
                nsaResults = nsaRankingTv.getNsaRankingResult(searchQuery);
            } else if (skill.equals(Constant.MIWEB_SKILL_ROUTER)) {
                nsaResults = nsaRankingRouter.getNsaRankingResult(searchQuery);
            } else if (skill.equals(Constant.MIWEB_SKILL_PHONE)){
                nsaResults = nsaRankingPhone.getNsaRankingResult(searchQuery);
            } else if (searchQuery.getSkill().equals(Constant.MIWEB_SKILL_REDMI)){
                nsaResults = nsaRankingPhone.getNsaRankingResult(searchQuery);
            } else if (searchQuery.getSkill().equals(Constant.MIWEB_SKILL_NOTEBOOK)){
                nsaResults = nsaRankingNb.getNsaRankingResult(searchQuery);
            } else if (searchQuery.getSkill().equals(Constant.MIWEB_SKILL_SIMCARD)){
                nsaResults = nsaRankingPhoneCard.getNsaRankingResult(searchQuery);
            }
        } else if (searchQuery.getGroupInfo().keySet().contains("新网银行")) {
            nsaResults = nsaRankingNewNet.getNsaRankingResult(searchQuery);
        } else if (searchQuery.getGroupInfo().keySet().contains("金融服务")) {
            nsaResults = nsaRankingMiBank.getRankingResult(searchQuery);
        }

        return Singleton.GSON.toJson(nsaResults);
    }
}
