package com.xl.wavenet.views.routeop;

import org.vaadin.lineawesome.LineAwesomeIconUrl;

import com.vaadin.flow.component.html.IFrame;
import com.vaadin.flow.component.orderedlayout.HorizontalLayout;
import com.vaadin.flow.component.orderedlayout.VerticalLayout;
import com.vaadin.flow.router.Menu;
import com.vaadin.flow.router.PageTitle;
import com.vaadin.flow.router.Route;

@PageTitle("Route Optimization")
@Route("routeop")
@Menu(order = 0, icon = LineAwesomeIconUrl.FILE)
public class RoutePage extends VerticalLayout {

	private static final long serialVersionUID = -7297379750003092414L;

	public RoutePage() {
		setSpacing(false);
		HorizontalLayout hr = new HorizontalLayout();
		IFrame frame = new IFrame(
				"https://app.powerbi.com/reportEmbed?reportId=3b512227-046b-46a3-a1c0-d377c1a85583&autoAuth=true&ctid=a1eae0da-f0d1-449d-8854-f54ddbda8711&navContentPaneEnabled=false");
		IFrame chat = new IFrame(
				"https://apps.powerapps.com/play/e/default-a1eae0da-f0d1-449d-8854-f54ddbda8711/a/20d86607-a78a-4ddf-8e06-06fe335b75ff?tenantId=a1eae0da-f0d1-449d-8854-f54ddbda8711&sourcetime=1732355351504&source=portal");
		hr.add(frame,chat);
		frame.setSizeFull();
		add(hr);
		hr.setSizeFull();
		setSizeFull();

	}

}
