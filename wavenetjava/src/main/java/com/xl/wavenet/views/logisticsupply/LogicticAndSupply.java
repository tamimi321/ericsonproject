package com.xl.wavenet.views.logisticsupply;

import org.vaadin.lineawesome.LineAwesomeIconUrl;

import com.vaadin.flow.component.html.IFrame;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.router.Menu;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

@PageTitle("Logictic And Supply Chain")
@Route("logisticandsupply")
@Menu(order = 3, icon = LineAwesomeIconUrl.GLOBE_SOLID)
public class LogicticAndSupply extends HorizontalLayout {

	private static final long serialVersionUID = -1876409253364505358L;

	public LogicticAndSupply() {
		setSpacing(false);
		HorizontalLayout hr = new HorizontalLayout();
		IFrame frame = new IFrame(
				"https://app.powerbi.com/reportEmbed?reportId=d21848ea-7e56-4a84-9e76-c2175e46f3ef&autoAuth=true&ctid=a1eae0da-f0d1-449d-8854-f54ddbda8711&navContentPaneEnabled=false");
		IFrame chat = new IFrame(
				"https://apps.powerapps.com/play/e/default-a1eae0da-f0d1-449d-8854-f54ddbda8711/a/20d86607-a78a-4ddf-8e06-06fe335b75ff?tenantId=a1eae0da-f0d1-449d-8854-f54ddbda8711&sourcetime=1732355351504&source=portal");
		hr.add(frame, chat);
		frame.setSizeFull();
		add(hr);
		hr.setSizeFull();
		setSizeFull();
	}

}