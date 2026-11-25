package vn.edu.dswinir;

import java.util.ArrayList;
import java.util.List;

/** Non-overlapping tiles. Edge padding replicates the last valid pixel. */
public final class Tiles {
    public static final int SIZE = 256;
    public static List<int[]> plan(int width, int height) {
        if (width < 1 || height < 1) throw new IllegalArgumentException("Empty image");
        List<int[]> result = new ArrayList<>();
        for (int y=0; y<height; y+=SIZE)
            for (int x=0; x<width; x+=SIZE)
                result.add(new int[]{x,y,Math.min(SIZE,width-x),Math.min(SIZE,height-y)});
        return result;
    }
    private Tiles() {}
}
