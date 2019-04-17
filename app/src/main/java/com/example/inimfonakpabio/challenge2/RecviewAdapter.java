package com.example.inimfonakpabio.challenge2;

import android.content.Context;
import android.support.v7.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class RecviewAdapter extends RecyclerView.Adapter<RecviewAdapter.ViewHolder> {

    Context mContext;
    List<String> mTexts;
    Map<String, Float> mMap;

    public RecviewAdapter(List<String> texts, Context con) {
        mTexts = texts;
        mContext = con;
    }

    public RecviewAdapter(Map<String, Float> map, Context con) {
        mContext = con;
        mMap = map;
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(mContext).inflate(R.layout.item_layout, parent, false);
        return new ViewHolder(v);
    }

    @Override
    public void onBindViewHolder(ViewHolder holder, int position) {
        if (mMap == null) {
            holder.textDisplay.setText(mTexts.get(position));
        } else {
            List keys = new ArrayList(mMap.keySet());
            Object label = keys.get(position);
            String text = String.format("%s: %4.2f",label.toString(),mMap.get(label));
            holder.textDisplay.setText( text );
        }
    }

    @Override
    public int getItemCount() {
        if (mMap == null) {
            return mTexts.size();
        } else {
            return mMap.size();
        }
    }

    public class ViewHolder extends RecyclerView.ViewHolder {
        TextView textDisplay;

        public ViewHolder(View itemView) {
            super(itemView);

            textDisplay = (TextView) itemView.findViewById(R.id.textDisplay);
        }
    }
}
