package vn.edu.dswinir;

import android.app.Activity;
import android.content.*;
import android.graphics.*;
import android.graphics.drawable.GradientDrawable;
import android.content.res.ColorStateList;
import android.view.*;
import android.net.Uri;
import android.os.*;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.*;
import java.util.function.BooleanSupplier;

public final class MainActivity extends Activity {
    private final ExecutorService worker=Executors.newSingleThreadExecutor();
    private final List<Button> buttons=new ArrayList<>();
    private final Map<Button,BooleanSupplier> prerequisites=new HashMap<>();
    private boolean busy, stoppable;
    private volatile boolean cancelRequested;
    private volatile Restorer activeRestorer;
    private Button cancelAction;
    private TextView readiness;
    private TextView status;
    private LinearLayout content;
    private final List<LinearLayout> pages=new ArrayList<>();
    private final List<Button> tabs=new ArrayList<>();
    private int activePage;
    private ProgressBar progress;
    private ScrollView workspace;
    private LinearLayout actionBar, resultCard;
    private Button restoreAction;
    private TextView actionHint;
    private final Handler ticker=new Handler(Looper.getMainLooper());
    private long jobStarted;
    private final Runnable tick=new Runnable() { public void run() {
        if (!busy) return;
        actionHint.setText((cancelRequested?"Đang dừng · Đã chờ ":"Đang xử lý · Đã chờ ")+((SystemClock.elapsedRealtime()-jobStarted)/1000)+" giây");
        ticker.postDelayed(this,1000);
    }};
    private static final int INK=0xff142b3b, MUTED=0xff536879, TEAL=0xff087e86, PAPER=0xfff2f6f8;
    private ImageView beforeView, afterView;
    private Spinner task;
    private Bitmap original, restored;
    private File restorationModel, detectionModel, segmentationModel;
    private TestSet dataset;
    private JSONObject report=new JSONObject();
    private static final int RESTORER=1, DETECTOR=2, SEGMENTER=3, IMAGE=4, DATASET=5, EXPORT=6;
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        restorationModel=new File(getFilesDir(),"restoration.tflite"); detectionModel=new File(getFilesDir(),"detection.tflite"); segmentationModel=new File(getFilesDir(),"segmentation.tflite");
        getWindow().setStatusBarColor(INK); getWindow().setNavigationBarColor(PAPER);
        LinearLayout shell=new LinearLayout(this); shell.setOrientation(LinearLayout.VERTICAL); shell.setBackgroundColor(PAPER); setContentView(shell);
        LinearLayout hero=new LinearLayout(this); hero.setOrientation(LinearLayout.VERTICAL); hero.setPadding(dp(22),dp(24),dp(22),dp(20)); hero.setBackgroundColor(INK); shell.addView(hero);
        TextView eyebrow=label("CÔNG CỤ KHÔI PHỤC ẢNH",11,0xff8edcdb); eyebrow.setLetterSpacing(.14f); hero.addView(eyebrow);
        TextView title=label("DSwinIR Studio",28,Color.WHITE); title.setTypeface(null,Typeface.BOLD); hero.addView(title);
        hero.addView(label("Làm rõ ảnh và so sánh kết quả trước, sau.",14,0xffc1d4de));
        LinearLayout navigation=new LinearLayout(this); navigation.setPadding(dp(16),dp(12),dp(16),dp(8)); shell.addView(navigation);
        String[] names={"Khôi phục","Đánh giá","Thiết lập"};
        for(int i=0;i<names.length;i++) {
            final int index=i; Button tab=new Button(this); tab.setText(names[i]); tab.setAllCaps(false); tab.setTextSize(13); tab.setMinWidth(0); tab.setMinimumWidth(0);
            LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(0,dp(44),1); lp.setMargins(dp(3),0,dp(3),0); navigation.addView(tab,lp);
            tab.setOnClickListener(v->selectPage(index)); tabs.add(tab);
        }
        ScrollView scroll=new ScrollView(this); workspace=scroll; scroll.setFillViewport(true); shell.addView(scroll,new LinearLayout.LayoutParams(-1,0,1));
        LinearLayout body=new LinearLayout(this); body.setOrientation(LinearLayout.VERTICAL); body.setPadding(dp(18),dp(8),dp(18),dp(28)); scroll.addView(body);
        content=body;
        LinearLayout sessionCard=card("TRẠNG THÁI","Bạn đang ở bước nào?");
        readiness=label("",12,MUTED); sessionCard.addView(readiness);
        status=label("Chọn một ảnh để bắt đầu.",14,INK); status.setTextIsSelectable(true); status.setPadding(0,dp(10),0,dp(4)); sessionCard.addView(status);
        progress=new ProgressBar(this,null,android.R.attr.progressBarStyleHorizontal); progress.setIndeterminate(true); progress.setIndeterminateTintList(ColorStateList.valueOf(TEAL)); progress.setVisibility(View.GONE); sessionCard.addView(progress,new LinearLayout.LayoutParams(-1,dp(4)));
        for(int i=0;i<3;i++) { LinearLayout page=new LinearLayout(this); page.setOrientation(LinearLayout.VERTICAL); body.addView(page); pages.add(page); }
        content=pages.get(0);
        LinearLayout photo=card("01 / NGUỒN ẢNH","Bắt đầu từ một bức ảnh");
        photo.addView(label("Chọn ảnh cần khôi phục và vấn đề của ảnh bên dưới.",13,MUTED));
        LinearLayout root=content; content=photo;
        button("Chọn ảnh từ thiết bị",()->pick(IMAGE,"image/*"));
        task=new Spinner(this); task.setBackground(round(PAPER,12)); task.setAdapter(new ArrayAdapter<>(this,android.R.layout.simple_spinner_dropdown_item,new String[]{"Ảnh bị nhiễu, có hạt","Ảnh có vệt mưa","Ảnh có sương mù","Ảnh bị nhòe","Ảnh thiếu sáng"})); content.addView(task,new LinearLayout.LayoutParams(-1,dp(50)));
        content.addView(label("Nếu chưa cài công cụ khôi phục, hãy mở mục Thiết lập trước.",12,MUTED));
        content=root;
        LinearLayout run=card("02 / XỬ LÝ","Chọn cách xử lý"); content=run;
        content.addView(label("Dùng cách thông thường, hoặc thử bộ xử lý đồ họa. Nếu thiết bị không hỗ trợ, ứng dụng tự chuyển về cách thông thường.",13,MUTED));

        button("Thử khôi phục bằng bộ xử lý đồ họa",()->restore(true),this::canRestore);
        content=root;
        LinearLayout result=card("03 / KẾT QUẢ","So sánh trước và sau"); resultCard=result; content=result;
        beforeView=imageView("TRƯỚC · Ảnh ban đầu"); afterView=imageView("SAU · Ảnh khôi phục");
        content.addView(label("Khi kiểm tra vật thể, các vùng nhận diện cũng xuất hiện trên hai ảnh này.",12,MUTED));
        button("Lưu ảnh và kết quả",this::export,()->restored!=null || report.length()>0);
        content=pages.get(1);
        LinearLayout vision=card("NHẬN DIỆN VẬT THỂ","Hiểu nội dung bức ảnh"); root=content; content=vision;
        content.addView(label("So sánh khả năng tìm vật thể trên hai ảnh. Chọn vẽ khung bao quanh hoặc tô vùng từng vật thể.",13,MUTED));
        button("Tìm và khoanh vật thể",()->preview(false),()->restored!=null && ready(detectionModel));
        button("Tô vùng từng vật thể",()->preview(true),()->restored!=null && ready(segmentationModel));
        content=root;
        LinearLayout batch=card("KIỂM TRA NHIỀU ẢNH","So sánh bằng điểm số"); content=batch;
        button("Chọn bộ ảnh kiểm tra (.zip)",()->pick(DATASET,"*/*"));
        button("Khôi phục tất cả ảnh trong bộ",this::restoreBatch,()->dataset!=null && ready(restorationModel));
        content.addView(label("Để chấm điểm, bộ ảnh phải kèm đáp án đúng: vị trí hoặc vùng của từng vật thể. Chưa có đáp án, bạn vẫn có thể khôi phục và xem ảnh.",13,MUTED));
        button("Chấm điểm tìm vật thể",()->evaluate(false),()->ready(restorationModel) && ready(detectionModel) && dataset!=null && dataset.hasGroundTruth(false));
        button("Chấm điểm tô vùng vật thể",()->evaluate(true),()->ready(restorationModel) && ready(segmentationModel) && dataset!=null && dataset.hasGroundTruth(true));
        content=root;
        LinearLayout speed=card("TỐC ĐỘ XỬ LÝ","Thiết bị xử lý nhanh đến đâu?"); content=speed;
        content.addView(label("Đây là phép đo chuyên sâu: chạy thử 5 lần và đo 50 lần. Trên máy ảo có thể mất hàng chục phút. Bạn có thể bấm Dừng xử lý để kết thúc sớm.",13,MUTED));
        button("Bắt đầu đo tốc độ",()->new android.app.AlertDialog.Builder(this).setTitle("Đo tốc độ chuyên sâu?").setMessage("Phép đo chạy thử 5 lần rồi đo 50 lần cho mỗi cách xử lý. Máy ảo có thể chậm trong thời gian dài. Để khôi phục một ảnh, hãy dùng nút Bắt đầu khôi phục.").setNegativeButton("Để sau",null).setPositiveButton("Đo tốc độ",(d,w)->benchmark()).show(),this::canRestore);
        content=pages.get(2);
        LinearLayout models=card("CÀI CÔNG CỤ","Chuẩn bị trước khi sử dụng"); content=models;
        content.addView(label("Chọn các tệp công cụ đi kèm ứng dụng. Bạn chỉ cần cài một lần cho mỗi công cụ.",13,MUTED));
        button("Cài công cụ khôi phục ảnh",()->pick(RESTORER,"*/*"));
        button("Cài công cụ khoanh vật thể",()->pick(DETECTOR,"*/*"));
        button("Cài công cụ tô vùng vật thể",()->pick(SEGMENTER,"*/*"));
        content.addView(label("LƯU Ý VỀ CHẤT LƯỢNG\nCông cụ đi kèm được huấn luyện để xóa vệt mưa. Kết quả với ảnh nhiễu, nhòe, có sương hoặc thiếu sáng chưa được bảo đảm.",13,MUTED));
        actionBar=new LinearLayout(this); actionBar.setOrientation(LinearLayout.VERTICAL); actionBar.setPadding(dp(18),dp(6),dp(18),dp(12)); actionBar.setBackgroundColor(Color.WHITE); actionBar.setElevation(dp(8)); shell.addView(actionBar);
        actionHint=label("Chọn ảnh để bắt đầu.",12,MUTED); actionBar.addView(actionHint);
        content=actionBar; button("Bắt đầu khôi phục",()->restore(false),this::canRestore); restoreAction=buttons.get(buttons.size()-1);
        cancelAction=new Button(this); cancelAction.setText("Dừng xử lý"); cancelAction.setAllCaps(false); cancelAction.setTextColor(TEAL); cancelAction.setOnClickListener(v->cancelJob()); actionBar.addView(cancelAction,new LinearLayout.LayoutParams(-1,dp(44)));
        content=body; selectPage(0); refreshActions();
        if (new File(getFilesDir(),"last-session/input.png").isFile()) loadSession();
    }
    private String taskName(String id) {
        switch(id) { case "denoise": return "Ảnh nhiễu"; case "derain": return "Ảnh có mưa"; case "dehaze": return "Ảnh có sương"; case "deblur": return "Ảnh nhòe"; case "lowlight": return "Ảnh thiếu sáng"; default: return id; }
    }
    private void saveSession() {
        if (original==null) return;
        try {
            File dir=new File(getFilesDir(),"last-session"); dir.mkdirs();
            new File(dir,"output.png").delete(); new File(dir,"report.json").delete();
            saveBitmap(original,new File(dir,"input.png"));
            if (restored!=null) saveBitmap(restored,new File(dir,"output.png")); else new File(dir,"output.png").delete();
            java.nio.file.Files.write(new File(dir,"report.json").toPath(),report.toString().getBytes(StandardCharsets.UTF_8));
        } catch (IOException e) { android.util.Log.w("DSwinIR","Cannot save last session",e); }
    }
    private void saveBitmap(Bitmap image,File file) throws IOException {
        File pending=new File(file.getPath()+".pending");
        try (OutputStream out=new FileOutputStream(pending)) {
            if (!image.compress(Bitmap.CompressFormat.PNG,100,out)) throw new IOException("Cannot save image");
        }
        java.nio.file.Files.move(pending.toPath(),file.toPath(),java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }
    private void loadSession() {
        job(()->{
            File dir=new File(getFilesDir(),"last-session");
            original=BitmapFactory.decodeFile(new File(dir,"input.png").getPath());
            restored=BitmapFactory.decodeFile(new File(dir,"output.png").getPath());
            if (original==null) throw new IOException("Không mở được ảnh đã lưu. Hãy chọn lại ảnh.");
            if (restored!=null && (restored.getWidth()!=original.getWidth() || restored.getHeight()!=original.getHeight())) restored=null;
            File saved=new File(dir,"report.json");
            if (saved.isFile()) report=new JSONObject(new String(java.nio.file.Files.readAllBytes(saved.toPath()),StandardCharsets.UTF_8));
            runOnUiThread(()->{ beforeView.setImageBitmap(original); afterView.setImageBitmap(restored); if(restored!=null) workspace.post(()->workspace.smoothScrollTo(0,pages.get(0).getTop()+resultCard.getTop()+afterView.getTop()-dp(28))); });
            show(restored==null?"Đã mở lại ảnh vừa chọn. Nhấn Bắt đầu khôi phục để tiếp tục.":"Đã mở lại kết quả đã lưu. Cuộn xuống xem ảnh trước và sau.");
        });
    }
    private int dp(int value) { return Math.round(value*getResources().getDisplayMetrics().density); }
    private GradientDrawable round(int color,int radius) {
        GradientDrawable d=new GradientDrawable(); d.setColor(color); d.setCornerRadius(dp(radius)); return d;
    }
    private TextView label(String text,int size,int color) {
        TextView t=new TextView(this); t.setText(text); t.setTextSize(size); t.setTextColor(color); t.setPadding(0,dp(3),0,dp(5)); return t;
    }
    private LinearLayout card(String caption,String title) {
        LinearLayout box=new LinearLayout(this); box.setOrientation(LinearLayout.VERTICAL); box.setPadding(dp(18),dp(16),dp(18),dp(18)); box.setBackground(round(Color.WHITE,20)); box.setElevation(dp(1));
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2); lp.setMargins(0,dp(3),0,dp(14)); content.addView(box,lp);
        TextView tag=label(caption,10,TEAL); tag.setTypeface(null,Typeface.BOLD); tag.setLetterSpacing(.1f); box.addView(tag);
        TextView heading=label(title,20,INK); heading.setTypeface(null,Typeface.BOLD); box.addView(heading); return box;
    }
    private void selectPage(int index) {
        activePage=index;
        if (actionBar!=null) actionBar.setVisibility(index==0 || busy?View.VISIBLE:View.GONE);
        for(int i=0;i<pages.size();i++) pages.get(i).setVisibility(i==index?View.VISIBLE:View.GONE);
        for(int i=0;i<tabs.size();i++) { tabs.get(i).setBackground(round(i==index?INK:Color.WHITE,12)); tabs.get(i).setTextColor(i==index?Color.WHITE:MUTED); }
    }
    private ImageView imageView(String caption) {
        content.addView(label(caption,11,MUTED));
        ImageView view=new ImageView(this); view.setAdjustViewBounds(true); view.setMinimumHeight(dp(140)); view.setBackground(round(PAPER,12)); view.setScaleType(ImageView.ScaleType.FIT_CENTER); view.setContentDescription(caption);
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2); lp.setMargins(0,dp(4),0,dp(16)); content.addView(view,lp); return view;
    }
    private void button(String label,Runnable action) { button(label,action,()->true); }
    private void button(String label,Runnable action,BooleanSupplier available) {
        Button b=new Button(this); b.setText(label); b.setAllCaps(false); b.setTextSize(14); b.setTypeface(null,Typeface.BOLD); b.setGravity(Gravity.CENTER); b.setMinHeight(dp(50)); b.setMinimumHeight(dp(50)); b.setPadding(dp(12),dp(10),dp(12),dp(10));
        b.setBackgroundTintList(null);
        android.graphics.drawable.StateListDrawable bg=new android.graphics.drawable.StateListDrawable();
        bg.addState(new int[]{-android.R.attr.state_enabled},round(0xffe9eef1,12));
        bg.addState(new int[]{android.R.attr.state_pressed},round(0xff07535b,12));
        boolean primary=label.equals("Bắt đầu khôi phục") || label.equals("Chọn ảnh từ thiết bị");
        bg.addState(new int[]{},round(primary?TEAL:0xffe7f3f3,12)); b.setBackground(bg);
        b.setTextColor(new ColorStateList(new int[][]{new int[]{-android.R.attr.state_enabled},new int[]{}},new int[]{0xff738493,primary?Color.WHITE:TEAL}));
        b.setOnClickListener(v->action.run()); buttons.add(b); prerequisites.put(b,available);
        LinearLayout.LayoutParams lp=new LinearLayout.LayoutParams(-1,-2); lp.setMargins(0,dp(8),0,dp(4)); content.addView(b,lp);
    }
    private static boolean ready(File file) { return file!=null && file.isFile() && file.length()>0; }
    private boolean canRestore() { return original!=null && ready(restorationModel); }
    private void refreshActions() {
        for (Button b:buttons) b.setEnabled(!busy && prerequisites.get(b).getAsBoolean());
        task.setEnabled(!busy);
        progress.setVisibility(busy?View.VISIBLE:View.GONE);
        cancelAction.setVisibility(busy && stoppable?View.VISIBLE:View.GONE);
        cancelAction.setEnabled(!cancelRequested);
        restoreAction.setText(busy?"Đang xử lý ảnh…":"Bắt đầu khôi phục");
        if (!busy) actionHint.setText(restored!=null?"Ảnh kết quả đã sẵn sàng ở phía trên.":original==null?"Chọn ảnh từ thiết bị ở phía trên.":ready(restorationModel)?"Ảnh đã chọn. Bấm nút bên dưới để khôi phục.":"Mở Thiết lập để cài công cụ khôi phục trước.");
        actionBar.setVisibility(activePage==0 || busy?View.VISIBLE:View.GONE);
        String image=original==null?"chưa chọn":original.getWidth()+"×"+original.getHeight();
        readiness.setText("Khôi phục ảnh: "+(ready(restorationModel)?"sẵn sàng":"cần cài công cụ")+" · Ảnh: "+image+
            "\nKhoanh vật thể: "+(ready(detectionModel)?"sẵn sàng":"chưa cài")+" · Tô vùng: "+(ready(segmentationModel)?"sẵn sàng":"chưa cài")+
            "\nBộ ảnh kiểm tra: "+(dataset==null?"chưa chọn":dataset.hasGroundTruth(false)?"có đáp án vị trí vật thể":"chưa có đáp án để chấm điểm"));
    }
    private void show(String text) { runOnUiThread(()->status.setText(text)); }
    private interface Job { void run() throws Exception; }
    private void checkCancelled() { if(cancelRequested) throw new CancellationException(); }
    private Restorer openRestorer(File file,boolean gpu) {
        checkCancelled();
        Restorer engine=new Restorer(file,gpu); activeRestorer=engine;
        if(cancelRequested) { engine.cancel(); engine.close(); throw new CancellationException(); }
        return engine;
    }
    private void cancelJob() {
        cancelRequested=true; cancelAction.setEnabled(false);
        show("Đang dừng xử lý. Bạn có thể cần chờ phần tính toán hiện tại kết thúc.");
        Restorer engine=activeRestorer; if(engine!=null) engine.cancel();
    }
    private void job(Job action) { job(false,action); }
    private void job(boolean allowStop,Job action) {
        if(busy) return;
        cancelRequested=false; activeRestorer=null; stoppable=allowStop;
        busy=true; jobStarted=SystemClock.elapsedRealtime(); refreshActions();
        ticker.removeCallbacks(tick); ticker.post(tick);
        workspace.smoothScrollTo(0,0);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        worker.execute(()->{
            try { action.run(); }
            catch (CancellationException e) { show("Đã dừng xử lý. Kết quả đã lưu trước đó vẫn được giữ; phép đo chưa hoàn tất không được tính thành kết quả."); }
            catch (Exception | LinkageError e) { show("Không thực hiện được: "+e.getMessage()); }
            finally { activeRestorer=null; runOnUiThread(()->{ busy=false; ticker.removeCallbacks(tick); getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); refreshActions(); }); }
        });
    }
    private void pick(int code,String mime) {
        Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT).setType(mime).addCategory(Intent.CATEGORY_OPENABLE); startActivityForResult(intent,code);
    }
    @Override protected void onActivityResult(int code,int result,Intent data) {
        super.onActivityResult(code,result,data); if (result!=RESULT_OK || data==null || data.getData()==null) return;
        Uri uri=data.getData();
        job(()->{
            if (code==IMAGE) {
                Bitmap decoded=ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(),uri), (d,info,s)->{
                    if ((long)info.getSize().getWidth()*info.getSize().getHeight()>24000000) throw new IllegalArgumentException("Ảnh quá lớn. Hãy chọn ảnh nhỏ hơn 24 triệu điểm ảnh.");
                    d.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                });
                if (decoded.getWidth()*(long)decoded.getHeight()>24000000) { decoded.recycle(); throw new IOException("Ảnh quá lớn. Hãy chọn ảnh nhỏ hơn 24 triệu điểm ảnh."); }
                original=decoded.copy(Bitmap.Config.ARGB_8888,false); decoded.recycle(); restored=null;
                runOnUiThread(()->{ beforeView.setImageBitmap(original); afterView.setImageDrawable(null); });
                report=new JSONObject(); saveSession(); show("Đã chọn ảnh "+original.getWidth()+"×"+original.getHeight()+". Nhấn «Bắt đầu khôi phục» để bắt đầu.");
            } else if (code==DATASET) {
                File folder=new File(getCacheDir(),"dataset-"+System.currentTimeMillis());
                dataset=TestSet.unpack(getContentResolver().openInputStream(uri),folder);
                report.remove("detection"); report.remove("segmentation"); report.remove("batch_restoration");
                show("Đã nhập "+dataset.images.length()+" ảnh. Có thể khôi phục trước. Chấm điểm cần đáp án đúng về vị trí hoặc vùng vật thể.");
            } else if (code==EXPORT) {
                try (OutputStream output=getContentResolver().openOutputStream(uri); ZipOutputStream zip=new ZipOutputStream(output)) {
                    zip.putNextEntry(new ZipEntry("report.json")); zip.write(report.toString(2).getBytes(StandardCharsets.UTF_8)); zip.closeEntry();
                    if (original!=null) { zip.putNextEntry(new ZipEntry("degraded.png")); original.compress(Bitmap.CompressFormat.PNG,100,zip); zip.closeEntry(); }
                    if (restored!=null) { zip.putNextEntry(new ZipEntry("restored.png")); restored.compress(Bitmap.CompressFormat.PNG,100,zip); zip.closeEntry(); }
                    File images=new File(getFilesDir(),"evaluation-images");
                    if ((report.has("detection") || report.has("segmentation") || report.has("batch_restoration")) && images.isDirectory()) for (File f:Objects.requireNonNull(images.listFiles())) {
                        zip.putNextEntry(new ZipEntry("evaluation/"+f.getName())); java.nio.file.Files.copy(f.toPath(),zip); zip.closeEntry();
                    }
                }
                show("Đã lưu ảnh và báo cáo trong tệp nén.");
            } else {
                File target=code==RESTORER?restorationModel:code==DETECTOR?detectionModel:segmentationModel;
                File pending=new File(getCacheDir(),"import.tflite");
                try (InputStream in=getContentResolver().openInputStream(uri); OutputStream out=new FileOutputStream(pending)) { copy(in,out,1024L*1024*1024); }
                if (code==RESTORER) try (Restorer check=new Restorer(pending,false)) { /* validate IO */ }
                else try (Yolo check=new Yolo(pending,code==SEGMENTER)) { /* validate task */ }
                java.nio.file.Files.copy(pending.toPath(),target.toPath(),java.nio.file.StandardCopyOption.REPLACE_EXISTING); pending.delete();
                if (code==RESTORER) { restored=null; report=new JSONObject(); saveSession(); runOnUiThread(()->afterView.setImageDrawable(null)); }
                show("Đã cài công cụ. "+(original==null?"Chọn ảnh để tiếp tục.":"Nhấn «Bắt đầu khôi phục» để bắt đầu."));
            }
        });
    }
    private static String sha(File file) throws Exception {
        MessageDigest md=MessageDigest.getInstance("SHA-256"); byte[] b=new byte[65536];
        try (InputStream in=new FileInputStream(file)) { int n; while ((n=in.read(b))!=-1) md.update(b,0,n); }
        StringBuilder s=new StringBuilder(); for (byte v:md.digest()) s.append(String.format(Locale.US,"%02x",v&255)); return s.toString();
    }
    private static void copy(InputStream in,OutputStream out,long limit) throws IOException {
        byte[] buffer=new byte[65536]; long total=0; int n;
        while ((n=in.read(buffer))!=-1) { total+=n; if (total>limit) throw new IOException("File exceeds size limit"); out.write(buffer,0,n); }
    }
    private void requireImage() { if (original==null) throw new IllegalStateException("Hãy chọn một ảnh trước."); }
    private void restore(boolean gpu) {
        final String degradation=new String[]{"denoise","derain","dehaze","deblur","lowlight"}[task.getSelectedItemPosition()];
        job(true,()->{
            requireImage(); show("Đang khôi phục ảnh…");
            String requested=gpu?"GPU":"CPU", error=null; long start=SystemClock.elapsedRealtimeNanos();
            try (Restorer engine=openRestorer(restorationModel,gpu)) { restored=engine.restore(original,(done,total)->show("Đang khôi phục ảnh · "+(done==total?"đã xử lý "+done:"đang xử lý "+(done+1))+"/"+total+" phần ảnh.\nMáy ảo có thể xử lý chậm. Bạn hãy chờ ảnh kết quả xuất hiện.")); }
            catch (RuntimeException | LinkageError e) {
                checkCancelled(); if(e instanceof CancellationException) throw e;
                if (!gpu) throw e;
                error=e.toString(); requested="CPU_FALLBACK";
                try (Restorer engine=openRestorer(restorationModel,false)) { restored=engine.restore(original,(done,total)->show("Đang khôi phục ảnh · "+(done==total?"đã xử lý "+done:"đang xử lý "+(done+1))+"/"+total+" phần ảnh.\nMáy ảo có thể xử lý chậm. Bạn hãy chờ ảnh kết quả xuất hiện.")); }
            }
            JSONObject r=new JSONObject().put("requested",gpu?"GPU":"CPU").put("executed",requested).put("gpuCoverage",gpu?"UNKNOWN_GPU_COVERAGE":"CPU_ONLY")
                .put("degradation",degradation).put("model_sha256",sha(restorationModel)).put("width",restored.getWidth()).put("height",restored.getHeight())
                .put("end_to_end_ms",(SystemClock.elapsedRealtimeNanos()-start)/1e6).put("gpu_error",error);
            report.put("restoration",r); saveSession(); runOnUiThread(()->{ afterView.setImageBitmap(restored); selectPage(0); workspace.post(()->workspace.smoothScrollTo(0,pages.get(0).getTop()+resultCard.getTop()+afterView.getTop()-dp(28))); }); show("Đã khôi phục "+restored.getWidth()+"×"+restored.getHeight()+String.format(Locale.US," trong %.1f giây.",r.getDouble("end_to_end_ms")/1000)+"\nCuộn xuống xem hai ảnh. Mở mục Đánh giá nếu muốn tìm hoặc tô vùng vật thể.");
        });
    }
    private String readGpuLogs(String since) {
        File log=new File(getCacheDir(),"gpu-runtime.log");
        try {
            java.lang.Process p=new ProcessBuilder("logcat","-d","-v","raw","--pid="+android.os.Process.myPid(),"-T",since,"tflite:V","*:S").redirectErrorStream(true).redirectOutput(log).start();
            if (!p.waitFor(3,TimeUnit.SECONDS)) { p.destroy(); return "Logcat unavailable"; }
            return new String(java.nio.file.Files.readAllBytes(log.toPath()),StandardCharsets.UTF_8);
        } catch (Exception e) { return "Logcat unavailable: "+e; }
    }
    private JSONObject measure(boolean gpu) throws Exception {
        String since=new SimpleDateFormat("MM-dd HH:mm:ss.SSS",Locale.US).format(new Date());
        JSONObject value=new JSONObject().put("requested",gpu?"GPU":"CPU").put("warmup",5).put("runs",50).put("unit","ms/tile 256x256; interpreter.run including output readback");
        try (Restorer engine=openRestorer(restorationModel,gpu)) {
            double[] samples=engine.benchmark(original,(done,total)->show("Đo "+(gpu?"bộ xử lý đồ họa":"bộ xử lý chính")+" · "+(done<5?"chạy thử "+(done+1)+"/5":done<55?"đang đo "+(done-4)+"/50":"đã đo 50/50")+". Có thể bấm Dừng xử lý.")); double sum=0; JSONArray raw=new JSONArray();
            for (double s:samples) { sum+=s; raw.put(s); }
            value.put("status","OK").put("mean_ms",sum/50).put("samples_ms",raw);
            String logs=gpu?readGpuLogs(since):"";
            int ops=ModelGraph.operatorCount(restorationModel);
            value.put("coverage",gpu?GpuEvidence.classify(logs,ops):"CPU_ONLY").put("original_single_graph_ops",ops).put("gpu_log",logs);
        } catch (CancellationException e) { throw e; } catch (RuntimeException | LinkageError e) { checkCancelled(); value.put("status","FAILED").put("error",e.toString()).put("coverage",gpu?"GPU_ATTEMPT_FAILED":"CPU_FAILED"); }
        return value;
    }
    private void benchmark() {
        job(true,()->{
            requireImage(); show("Đang đo bộ xử lý chính: chạy thử 5 lần, đo 50 lần trên phần ảnh đầu tiên…"); JSONObject cpu=measure(false);
            checkCancelled();
            show("Đã đo bộ xử lý chính. Đang chạy thử và đo bộ xử lý đồ họa…"); JSONObject gpu=measure(true);
            checkCancelled();
            JSONObject b=new JSONObject().put("device",Build.MANUFACTURER+" "+Build.MODEL).put("android_api",Build.VERSION.SDK_INT).put("model_sha256",sha(restorationModel)).put("cpu_threads",Restorer.cpuThreads())
                .put("fp32_io",true).put("cpu_xnnpack",false).put("gpu_precision_loss_allowed",false).put("cpu",cpu).put("gpu_attempt",gpu)
                .put("note","FULL/PARTIAL/CPU_ONLY chỉ xác nhận từ log runtime. UNKNOWN cần profiling; trường hợp không xuất hiện trên thiết bị này là N/A.");
            JSONObject scenarios=new JSONObject().put("FULL_GPU","N/A").put("PARTIAL_GPU","N/A").put("GPU_UNAVAILABLE_CPU","N/A");
            String coverage=gpu.optString("coverage");
            if (coverage.equals("FULL_GPU") || coverage.equals("PARTIAL_GPU")) scenarios.put(coverage,gpu);
            if ((gpu.optString("status").equals("FAILED") || coverage.equals("CPU_ONLY")) && cpu.optString("status").equals("OK"))
                scenarios.put("GPU_UNAVAILABLE_CPU",new JSONObject().put("cpu_baseline",cpu).put("gpu_attempt",gpu));
            b.put("scenarios",scenarios); report.put("benchmark",b);
            String cpuTime=cpu.has("mean_ms")?String.format(Locale.US,"%.3f ms/phần ảnh",cpu.getDouble("mean_ms")):"không chạy được";
            String gpuTime=gpu.has("mean_ms")?String.format(Locale.US,"%.3f ms/phần ảnh",gpu.getDouble("mean_ms")):"không chạy được";
            show("Thời gian trung bình trên một phần ảnh 256×256 (thấp hơn là nhanh hơn)\nBộ xử lý chính (CPU): "+cpuTime+"\nBộ xử lý đồ họa (GPU): "+gpuTime+
                "\nChạy toàn bộ bằng đồ họa: "+(coverage.equals("FULL_GPU")?gpuTime:"chưa ghi nhận")+
                "\nChạy kết hợp đồ họa và bộ xử lý chính: "+(coverage.equals("PARTIAL_GPU")?gpuTime:"chưa ghi nhận")+
                "\nKhông dùng được đồ họa, chuyển về bộ xử lý chính: "+(scenarios.opt("GPU_UNAVAILABLE_CPU") instanceof JSONObject?cpuTime:"chưa ghi nhận")+
                (coverage.equals("UNKNOWN_GPU_COVERAGE")?"\nChưa xác định được phần việc do đồ họa xử lý.":"")+
                "\nĐã chạy thử 5 lần và đo 50 lần cho mỗi cách chạy thành công. Chưa gồm thời gian chuẩn bị và ghép ảnh. Lưu kết quả để xem số liệu chi tiết.");
        });
    }
    private void preview(boolean segment) {
        job(()->{
            requireImage(); if (restored==null) throw new IllegalStateException("Khôi phục ảnh trước");
            show(segment?"Đang tìm và tô vùng vật thể trên hai ảnh…":"Đang tìm và khoanh vật thể trên hai ảnh…");
            try (Yolo yolo=new Yolo(segment?segmentationModel:detectionModel,segment)) {
                List<Instance> before=yolo.predict(original,"preview"), after=yolo.predict(restored,"preview");
                Bitmap bv=Yolo.draw(original,before), av=Yolo.draw(restored,after);
                runOnUiThread(()->{ beforeView.setImageBitmap(bv); afterView.setImageBitmap(av); selectPage(0); workspace.post(()->workspace.smoothScrollTo(0,pages.get(0).getTop()+resultCard.getTop()+afterView.getTop()-dp(28))); });
                show("Đã "+(segment?"tô vùng":"khoanh")+" vật thể. Cuộn xuống xem hai ảnh.\nĐây là kết quả xem trước, chưa phải điểm đánh giá. Muốn chấm điểm, hãy chọn bộ ảnh có đáp án đúng.");
            }
        });
    }
    private void evaluate(boolean segment) {
        job(true,()->{
            if (dataset==null) throw new IllegalStateException("Để chấm điểm, hãy chọn bộ ảnh kiểm tra có đáp án đúng. Với một ảnh, hãy khôi phục rồi dùng chức năng khoanh hoặc tô vùng vật thể.");
            for (int i=0;i<dataset.images.length();i++)
                if (dataset.images.getJSONObject(i).optString("annotation_status").equals("pending"))
                    throw new IllegalStateException("Bộ ảnh chưa có đáp án đúng. Bạn có thể khôi phục và xem ảnh, nhưng chưa thể chấm điểm.");
            Map<String,List<Instance>> gt=new TreeMap<>(), before=new TreeMap<>(), after=new TreeMap<>(); Map<String,Integer> counts=new TreeMap<>();
            JSONArray samples=new JSONArray();
            report.remove("detection"); report.remove("segmentation"); report.remove("batch_restoration");
            File folder=new File(getFilesDir(),"evaluation-images"); folder.mkdirs();
            for (File f:Objects.requireNonNull(folder.listFiles())) f.delete();
            File yoloFile=segment?segmentationModel:detectionModel;
            try (Restorer engine=openRestorer(restorationModel,false); Yolo yolo=new Yolo(yoloFile,segment)) {
                for (int i=0;i<dataset.images.length();i++) {
                    checkCancelled();
                    JSONObject row=dataset.images.getJSONObject(i); String name=row.getString("degradation"), id=row.getString("id");
                    JSONObject sample=new JSONObject(row.toString()).put("input_sha256",sha(dataset.file(row.getString("input"))));
                    samples.put(sample);
                    show("Đánh giá "+(i+1)+"/"+dataset.images.length()+" · "+taskName(name)+" · "+(segment?"tô vùng vật thể":"khoanh vật thể"));
                    File sourceFile=dataset.file(row.getString("input"));
                    BitmapFactory.Options bounds=new BitmapFactory.Options(); bounds.inJustDecodeBounds=true;
                    BitmapFactory.decodeFile(sourceFile.getPath(),bounds);
                    if ((long)bounds.outWidth*bounds.outHeight>24000000) throw new IOException("Ảnh >24 megapixel");
                    Bitmap source=BitmapFactory.decodeFile(sourceFile.getPath());
                    if (source==null) throw new IOException("Không đọc được ảnh "+id);
                    if ((long)source.getWidth()*source.getHeight()>24000000) { source.recycle(); throw new IOException("Ảnh >24 megapixel"); }
                    try {
                        List<Instance> truth=TestSet.annotations(row,source.getWidth(),source.getHeight(),segment);
                        gt.computeIfAbsent(name,k->new ArrayList<>()).addAll(truth);
                        List<Instance> bp=yolo.predict(source,id);
                        before.computeIfAbsent(name,k->new ArrayList<>()).addAll(bp);
                        sample.put("before_predictions",predictions(bp));
                        Bitmap result=engine.restore(source);
                        try {
                            List<Instance> ap=yolo.predict(result,id);
                            after.computeIfAbsent(name,k->new ArrayList<>()).addAll(ap);
                            sample.put("after_predictions",predictions(ap));
                            try (OutputStream out=new FileOutputStream(new File(folder,String.format(Locale.US,"%04d-%s.png",i,name)))) { result.compress(Bitmap.CompressFormat.PNG,100,out); }
                        } finally { result.recycle(); }
                    } finally { source.recycle(); }
                    counts.put(name,counts.getOrDefault(name,0)+1);
                }
            }
            JSONObject metrics=new JSONObject();
            for (String name:new TreeSet<>(TestSet.TASKS)) {
                if (!gt.containsKey(name) || gt.get(name).isEmpty()) { metrics.put(name,new JSONObject().put("status","NO_GROUND_TRUTH_OR_IMAGES")); continue; }
                double[] b=MapMetric.compute(gt.get(name),before.get(name),segment), a=MapMetric.compute(gt.get(name),after.get(name),segment);
                metrics.put(name,new JSONObject().put("images",counts.get(name)).put("before_mAP50",b[0]).put("after_mAP50",a[0]).put("delta_mAP50",a[0]-b[0])
                    .put("before_mAP50_95",b[1]).put("after_mAP50_95",a[1]).put("delta_mAP50_95",a[1]-b[1]));
            }
            JSONObject result=new JSONObject().put("task",segment?"instance_segmentation":"object_detection").put("restoration_model_sha256",sha(restorationModel)).put("yolo_model_sha256",sha(yoloFile))
                .put("protocol","101-point AP; IoU .50:.05:.95; conf .001; class-aware NMS .7; max300; GT-present classes; no crowd/ignore; polygon raster masks")
                .put("restoration_backend","CPU").put("samples",samples).put("results",metrics);
            report.put(segment?"segmentation":"detection",result);
            StringBuilder summary=new StringBuilder("Điểm "+(segment?"tô vùng":"tìm")+" vật thể trước → sau khôi phục (mAP, cao hơn là tốt hơn):\n");
            for (String name:new TreeSet<>(TestSet.TASKS)) {
                JSONObject m=metrics.getJSONObject(name); summary.append(taskName(name)).append(": ");
                if (!m.has("before_mAP50_95")) summary.append("chưa đủ ảnh hoặc đáp án\n");
                else summary.append(String.format(Locale.US,"%.1f → %.1f / 100\n",100*m.getDouble("before_mAP50_95"),100*m.getDouble("after_mAP50_95")));
            }
            show(summary+"Lưu kết quả để xem báo cáo chi tiết.");
        });
    }
    private void restoreBatch() {
        job(true,()->{
            if (dataset==null) throw new IllegalStateException("Chọn bộ ảnh kiểm tra (.zip) trước");
            report.remove("detection"); report.remove("segmentation"); report.remove("batch_restoration");
            File folder=new File(getFilesDir(),"evaluation-images"); folder.mkdirs();
            for (File f:Objects.requireNonNull(folder.listFiles())) f.delete();
            report.remove("detection"); report.remove("segmentation");
            JSONArray samples=new JSONArray();
            try (Restorer engine=openRestorer(restorationModel,false)) {
                for (int i=0;i<dataset.images.length();i++) {
                    checkCancelled();
                    JSONObject row=dataset.images.getJSONObject(i);
                    show("Khôi phục "+(i+1)+"/"+dataset.images.length()+" · "+taskName(row.getString("degradation")));
                    File input=dataset.file(row.getString("input"));
                    BitmapFactory.Options bounds=new BitmapFactory.Options(); bounds.inJustDecodeBounds=true;
                    BitmapFactory.decodeFile(input.getPath(),bounds);
                    if ((long)bounds.outWidth*bounds.outHeight>24000000) throw new IOException("Ảnh >24 megapixel");
                    Bitmap source=BitmapFactory.decodeFile(input.getPath());
                    if (source==null) throw new IOException("Không đọc được ảnh test");
                    try {
                        long start=SystemClock.elapsedRealtimeNanos();
                        Bitmap result=engine.restore(source);
                        try {
                            double elapsed=(SystemClock.elapsedRealtimeNanos()-start)/1e6;
                            String filename=String.format(Locale.US,"%04d-%s.png",i,row.getString("degradation"));
                            try (OutputStream out=new FileOutputStream(new File(folder,filename))) { result.compress(Bitmap.CompressFormat.PNG,100,out); }
                            samples.put(new JSONObject(row.toString()).put("input_sha256",sha(input)).put("output",filename)
                                .put("width",result.getWidth()).put("height",result.getHeight()).put("restore_ms",elapsed));
                            if (i==dataset.images.length()-1) {
                                original=source.copy(Bitmap.Config.ARGB_8888,false); restored=result.copy(Bitmap.Config.ARGB_8888,false);
                                runOnUiThread(()->{ beforeView.setImageBitmap(original); afterView.setImageBitmap(restored); });
                            }
                        } finally { result.recycle(); }
                    } finally { source.recycle(); }
                }
            }
            report.put("batch_restoration",new JSONObject().put("model_sha256",sha(restorationModel)).put("backend","CPU")
                .put("timing","full image restoration; excludes shared interpreter initialization").put("samples",samples)
                .put("mAP","NOT_EVALUATED_WITHOUT_GROUND_TRUTH"));
            show("Đã khôi phục "+samples.length()+" ảnh, giữ kích thước gốc. Đang hiển thị ảnh cuối. Chọn Lưu ảnh và kết quả để lưu tất cả ảnh cùng báo cáo.\nChưa thể chấm điểm vì bộ ảnh chưa có đáp án đúng.");
        });
    }
    private static JSONArray predictions(List<Instance> values) throws JSONException {
        JSONArray out=new JSONArray();
        for (Instance p:values) {
            JSONArray box=new JSONArray(); for (float x:p.box) box.put(x);
            JSONObject item=new JSONObject().put("class_id",p.cls).put("score",p.score).put("bbox",box);
            if (p.mask!=null) {
                JSONArray runs=new JSONArray(); int start=p.mask.nextSetBit(0);
                while (start>=0) { int end=p.mask.nextClearBit(start); runs.put(start); runs.put(end-start); start=p.mask.nextSetBit(end); }
                item.put("mask_row_major_start_length",runs);
            }
            out.put(item);
        }
        return out;
    }
    private void export() {
        Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT).setType("application/zip").addCategory(Intent.CATEGORY_OPENABLE).putExtra(Intent.EXTRA_TITLE,"dswinir-report.zip"); startActivityForResult(intent,EXPORT);
    }
    @Override public void onDestroy() { ticker.removeCallbacks(tick); if(busy && stoppable) cancelJob(); worker.shutdown(); super.onDestroy(); }
}
