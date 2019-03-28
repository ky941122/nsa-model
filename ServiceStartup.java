package com.xiaomi.chatbot.services.nsasearch;

import com.networknt.server.StartupHookProvider;
import com.xiaomi.chatbot.common.MessageQueue;
//import com.xiaomi.chatbot.services.essearch.model.SearchDataUpdate;
import com.xiaomi.chatbot.services.common.zookeeper.ZkRegister;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by guoxiang <guoxiang1@xiaomi.com> on 07/31/18.
 */

@Slf4j
public class ServiceStartup implements StartupHookProvider {

    private static final String TOPIC_NAME = "ics-nsa-update";
    private static final String GROUP_NAME = "icsNsa";
    private static final String CLIENT_NAME = "ics-nsa-search";
    ZkRegister zkRegister;

    public void onStartup() {
        log.debug("Nsa Startup.");

        // 启动数据更新
        //MessageQueue.getInstance().createConsumer(TOPIC_NAME, false, GROUP_NAME, CLIENT_NAME, new SearchDataUpdate());
        try {
            zkRegister = new ZkRegister("ics-nsa-search", "8352");
        }catch(Throwable e){
            log.error("fail to register zk: {}", e);
        }
    }
}
