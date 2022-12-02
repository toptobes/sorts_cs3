import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

@SuppressWarnings("MismatchedReadAndWriteOfArray")
public class Sort {
    public static final int N = 1_000_000_000;

    public static void main(String[] args) {
        long t1, t2;
        int[] nums = new int[N];

        for (int i = 0; i < N; i++) {
            nums[i] = ThreadLocalRandom.current().nextInt(0, Integer.MAX_VALUE);
        }

        System.gc();

        t1 = System.currentTimeMillis();
        Arrays.parallelSort(nums);
        t2 = System.currentTimeMillis();
        System.out.println(t2 - t1);
    }
}
