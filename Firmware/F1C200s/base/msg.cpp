#include "msg.h"

message_queue g_worker;
message_queue g_uart;

void SendGlobalMsg(int type, int data1, int data2, int data3)
{
    MSG* msg = (MSG*)message_queue_message_alloc(&g_worker);
    if (msg == 0)
        return;
    msg->type = type;
    msg->data1 = data1;
    msg->data2 = data2;
    msg->data3 = data3;
    message_queue_write(&g_worker, (MSG*)msg);
}
