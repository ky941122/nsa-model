package com.xiaomi.chatbot.services.nsasearch;

import com.networknt.server.HandlerProvider;
import com.xiaomi.chatbot.services.common.handler.HealthHandler;
import com.xiaomi.chatbot.services.common.handler.PerfCounterHandler;
import com.xiaomi.chatbot.services.nsasearch.handler.NsaSearchHandler;
import io.undertow.Handlers;
import io.undertow.server.HttpHandler;
import io.undertow.util.Methods;

/**
 * Created by Hunt Tang <tangmingming@xiaomi.com> on 6/17/17.
 */
public class PathHandlerProvider implements HandlerProvider {
    @Override
    public HttpHandler getHandler() {
        return Handlers.routing()
                .add(Methods.GET, "/health", new HealthHandler())
                .add(Methods.POST, "/ics/nsa-search", new PerfCounterHandler(new NsaSearchHandler(), true, "nsasearch"));
    }
}
