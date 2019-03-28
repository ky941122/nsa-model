package com.xiaomi.chatbot.services.nsasearch.handler;

import com.xiaomi.chatbot.services.common.handler.BaseHttpHandler;
import com.xiaomi.chatbot.services.common.wrapper.ics.SearchQuery;
import com.xiaomi.chatbot.services.nsasearch.controller.NsaSearchController;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public class NsaSearchHandler extends BaseHttpHandler<SearchQuery> {
    private static NsaSearchController nsaSearchController = new NsaSearchController();

    @Override
    public String handleRequest(Map params) {
        return null;
    }

    @Override
    public String handleRequest(SearchQuery searchQuery) {
        return nsaSearchController.getResult(searchQuery);
    }

    @Override
    public String checkParam(SearchQuery searchQuery) {
        if (StringUtils.isBlank(searchQuery.getQuery())) {
            return " Got a blank search query: %s" + searchQuery.toString();
        }
        return "";
    }
}
