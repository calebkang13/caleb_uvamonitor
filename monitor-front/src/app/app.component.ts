import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
	selector: 'app-root',
	templateUrl: './app.component.html',
	styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

	title = 'Building Search Engine'; 
	chart = true;
	form = false;
	
	constructor(private http: HttpClient) {}

	ngOnInit(): void {
	}

	set_screen(): void {
		if (this.chart === true) {
			this.chart = false;
			this.form = true;
		} else {
			this.chart = true;
			this.form = false;
		}
	}
}
