import java.util.*;

class MessageQueue {
    private final Queue<String> queue = new LinkedList<>();

    public synchronized void produce(String message)
            throws InterruptedException {
        queue.add(message);
        System.out.println("Produced: " + message);
        notifyAll();
    }

    public synchronized String consume()
            throws InterruptedException {
        while (queue.isEmpty())
            wait();

        String message = queue.remove();
        System.out.println("Consumed: " + message);
        return message;
    }
}

public class ChatDemo {
    public static void main(String[] args) throws InterruptedException {
        MessageQueue q = new MessageQueue();

        Thread consumer = new Thread(() -> {
            try {
                System.out.println("Received: " + q.consume());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        });

        consumer.start();
        Thread.sleep(100);
        q.produce("Hello");
        consumer.join();
    }
}