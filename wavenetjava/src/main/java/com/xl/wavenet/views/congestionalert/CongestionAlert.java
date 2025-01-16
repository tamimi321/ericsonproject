package com.xl.wavenet.views.congestionalert;

import com.vaadin.flow.component.Key;
import com.vaadin.flow.component.button.Button;
import com.vaadin.flow.component.html.IFrame;
import com.vaadin.flow.component.notification.Notification;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.textfield.TextField;
import com.vaadin.flow.router.Menu;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;
import org.vaadin.lineawesome.LineAwesomeIconUrl;

@PageTitle("Congestion Alerts")
@Route("congestionalrt")
@Menu(order = 1, icon = LineAwesomeIconUrl.GLOBE_SOLID)
public class CongestionAlert extends HorizontalLayout {

    private static final long serialVersionUID = -1876409253364505358L;

    public CongestionAlert() {
		setSpacing(false);
		HorizontalLayout hr = new HorizontalLayout();
		IFrame frame = new IFrame(
				"https://app.powerbi.com/reportEmbed?reportId=d0aa6ea3-afde-45db-8c19-452dfbbac3ee&autoAuth=true&ctid=a1eae0da-f0d1-449d-8854-f54ddbda8711&navContentPaneEnabled=false");
		IFrame chat = new IFrame(
				"https://apps.powerapps.com/play/e/default-a1eae0da-f0d1-449d-8854-f54ddbda8711/a/20d86607-a78a-4ddf-8e06-06fe335b75ff?tenantId=a1eae0da-f0d1-449d-8854-f54ddbda8711&sourcetime=1732355351504&source=portal");
		hr.add(frame,chat);
		frame.setSizeFull();
		add(hr);
		hr.setSizeFull();
		setSizeFull();
    }

}
