package vn.edu.dswinir;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;

/** Read only the schema's Model.subgraphs and SubGraph.operators vectors.
 * Refuse control-flow/multiple subgraphs: a single replacement count would be
 * insufficient evidence of whole-model delegation in those cases.
 */
public final class ModelGraph {
    private static int field(ByteBuffer b,int table,int id) {
        int v=table-b.getInt(table), offset=4+2*id;
        if (offset+2>Short.toUnsignedInt(b.getShort(v))) return 0;
        int relative=Short.toUnsignedInt(b.getShort(v+offset));
        return relative==0?0:table+relative;
    }
    private static int target(ByteBuffer b,int pos) { return pos+b.getInt(pos); }
    public static int operatorCount(File file) {
        try (FileInputStream in=new FileInputStream(file)) {
            ByteBuffer b=in.getChannel().map(FileChannel.MapMode.READ_ONLY,0,file.length()).order(ByteOrder.LITTLE_ENDIAN);
            int root=b.getInt(0), f=field(b,root,2); if (f==0) return -1;
            int subgraphs=target(b,f); if (b.getInt(subgraphs)!=1) return -1;
            int graph=target(b,subgraphs+4), ops=field(b,graph,3);
            return ops==0?0:b.getInt(target(b,ops));
        } catch (IOException | RuntimeException e) { return -1; }
    }
    private ModelGraph() {}
}
