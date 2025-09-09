  {Part of Plot_Rsc Unit}
   {Plotting Vector Dumps / 2-D Stress/ Velocity Vectors}


     PROCEDURE Pl_StressLeg (Dmp : Word);
       CONST
         Hinset = 200;
       VAR
         Vstep  : Single;
         Tstr    : String;
       BEGIN
         Pl_SetWindow (Pl_LegWind);
         Vstep := Pl_LegWind.Sz[2] / 2;
         WITH DumpMap^ [Dmp] DO
           BEGIN
             Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
             Pl_SetTextJustify (CenterText, CenterText);
             Pl_SelectColour (7);
             Pl_MoveTo ( Pl_LegWind.Sz[1] DIV 2, TRUNC(Vstep * (1+ 8.5/10))  );
             Pl_OutText ('PRINCIPLE STRESS');
             Pl_MoveTo ( Pl_LegWind.Sz[1] DIV 2, TRUNC(Vstep * (1+7.5/10))  );
             Pl_OutText ('AXES - [' + Strng (Dmp,0) + ']');
             Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
             Pl_SetTextJustify (LeftText, CenterText);
             Pl_SelectColour (PlPen_Geom);
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+6/10))  );
             IF (Dim3)
                THEN Pl_OutText ('  3-D Model')
                ELSE Pl_OutText ('  2-D Model');
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+5/10))  );
             Pl_OutText ('Time    = ' + Strng (Time,$0300) );
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+4/10))  );
             Str (MaxVal : 12, Tstr);
             Pl_OutText ('Max Str = ' + Tstr);
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+3/10))  );
             Str (Pl_1cm / Pl_VectSc : 12, Tstr);
             Pl_OutText ('1CM : ' + Tstr + ' Pa');
             Pl_SelectColour (PlPen_CStress);
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+2/10))  );
             Pl_OutText ('--  : Compr. Stress');
             Pl_SelectColour (PlPen_TStress);
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+1/10))  );
             Pl_OutText ('==  : Tens.  Stress');
           END; {With}
         Pl_ResetViewPort;
       END; {Pl_StressLeg}


     PROCEDURE Pl_VectLeg (Dmp : Word);
       CONST
         Hinset = 200;
       VAR
         Vstep  : Single;
         Tstr    : String;
       BEGIN
         Pl_SetWindow (Pl_LegWind);
         Vstep := Pl_LegWind.Sz[2] / 2;
         WITH DumpMap^ [Dmp] DO
           BEGIN
             Pl_SetTextStyle (SmallFont, HorizDir, FontSz);
             Pl_SetTextJustify (CenterText, CenterText);
             Pl_SelectColour (7);
             Pl_MoveTo ( Pl_LegWind.Sz[1] DIV 2, TRUNC(Vstep * (1+ 8.5/10))  );
             Pl_OutText ('VELOCITY VECTORS');
             Pl_MoveTo ( Pl_LegWind.Sz[1] DIV 2, TRUNC(Vstep * (1+7.5/10))  );
             Pl_OutText ('[' + Strng (Dmp,0) + ']');
             Pl_SetTextStyle (SmallFont, HorizDir, FontSz-1);
             Pl_SetTextJustify (LeftText, CenterText);
             Pl_SelectColour (PlPen_Geom);
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+6/10))  );
             IF (Dim3)
                THEN Pl_OutText ('  3-D Model')
                ELSE Pl_OutText ('  2-D Model');
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+5/10))  );
             Pl_OutText ('Time    = ' + Strng (Time,$0300) );
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+4/10))  );
             Str (MaxVal : 12, Tstr);
             Pl_OutText ('Max Vel = ' + Tstr);
             Pl_MoveTo ( Hinset, TRUNC(Vstep * (1+3/10))  );
             Str (Pl_1cm / Pl_VectSc : 12, Tstr);
             Pl_OutText ('1CM : ' + Tstr + ' m/s');
           END; {With}
         Pl_ResetViewPort;
       END; {Pl_VectLeg}


       PROCEDURE Pl_StressAxes (Dmp : Word);
         VAR
           xPix, yPix, ix, iy : Integer;
           HzOff, VtOff       : Integer;
           Xblk, Yblk         : LongInt;
           Maj_len,  Min_len, Angle  : Single;

         BEGIN
           Pl_SetLineType (1);
           Pl_SetWindow  (Pl_SnapWind);
           WITH DumpMap^ [Dmp] DO
           WITH Scale2D DO
             BEGIN
               Xblk  := (Xstp * Xqty * Pl_Xpx) DIV QuickX;
               Yblk  := (Ystp * Yqty * Pl_Ypx) DIV QuickY;
               HzOff := Pl_Xoff + (Xst*Pl_Xpx) DIV Xlen;
               VtOff := Pl_Yoff + (Yst*Pl_Ypx) DIV Ylen;
               IF (Sets.VectSet)
                  THEN Pl_VectSc := Pl_1CM * Sets.VectSc
                  ELSE Pl_VectSc := Xblk / Xlen * Vzoom / MaxVal / VectZoom;;

               {Draw Compressive Stresses}
               Pl_SelectColour (PlPen_CStress);
               FOR ix := 1 TO QuickX DO
                 BEGIN
                   xPix := HzOff + ( (ix-1)*Xblk) DIV Xlen;
                   FOR iy := 1 TO QuickY DO
                     BEGIN
                       yPix := VtOff + ( (iy-1)*Yblk) DIV Ylen;
                       WITH Dump^ [ix, iy] DO
                         BEGIN
                           Maj_len  := IROUND (Pl_VectSc * Major);
                           Min_len  := IROUND (Pl_VectSc * Minor);
                           Angle := Ang;
                         END; {With}
                       IF (Maj_len < 0)
                          THEN Pl_CentreBar (xPix, yPix, Maj_len, Angle);
                       IF (Min_len < 0)
                          THEN Pl_CentreBar (xPix, yPix, Min_len, (Angle + PI/2));
                     END; {For iy}
                 END; {For ix - Compressive Stresses}

               {Draw Tensile Stresses}
               Pl_SelectColour (PlPen_TStress);
               FOR ix := 1 TO QuickX DO
                 BEGIN
                   xPix := HzOff + ( (ix-1)*Xblk) DIV Xlen;
                   FOR iy := 1 TO QuickY DO
                     BEGIN
                       yPix := VtOff + ( (iy-1)*Yblk) DIV Ylen;
                       WITH Dump^ [ix, iy] DO
                         BEGIN
                           Maj_len  := IROUND (Pl_VectSc * Major);
                           Min_len  := IROUND (Pl_VectSc * Minor);
                           Angle := Ang;
                         END; {With}
                       IF (Maj_len >= 0)
                          THEN Pl_CentreBar (xPix, yPix, Maj_len, Angle);
                       IF (Min_len >= 0)
                          THEN Pl_CentreBar (xPix, yPix, Min_len, (Angle + PI/2));
                     END; {For iy}
                 END; {For ix - Tensile Stresses}
             END; {With}
         END; {Pl_StressAxes}


       PROCEDURE Pl_VelVects (Dmp : Word);
         VAR
           xPix, yPix, ix, iy : Integer;
           HzOff, VtOff       : Integer;
           Xblk, Yblk         : LongInt;
           Hz_len, Vt_len     : Integer;
         BEGIN
           Pl_SetLineType (1);
           Pl_SetWindow  (Pl_SnapWind);
           WITH DumpMap^ [Dmp] DO
           WITH Scale2D DO
             BEGIN
               Xblk  := (Xstp * Xqty * Pl_Xpx) DIV QuickX;
               Yblk  := (Ystp * Yqty * Pl_Ypx) DIV QuickY;
               HzOff := Pl_Xoff + (Xst*Pl_Xpx) DIV Xlen;
               VtOff := Pl_Yoff + (Yst*Pl_Ypx) DIV Ylen;
               IF (Sets.VectSet)
                  THEN Pl_VectSc := Pl_1CM * Sets.VectSc
                  ELSE Pl_VectSc := Xblk / Xlen * Vzoom / MaxVal / VectZoom;

               {Draw Vector lines}
               Pl_SetColor (PlPen_Vect);
               FOR ix := 1 TO QuickX DO
                 BEGIN
                   xPix := HzOff + ( (ix-1)*Xblk) DIV Xlen;
                   FOR iy := 1 TO QuickY DO
                     BEGIN
                       yPix := VtOff + ( (iy-1)*Yblk) DIV Ylen;
                       WITH Dump^ [ix, iy] DO
                         BEGIN
                           Hz_len  := IROUND (Pl_VectSc * V[X]);
                           Vt_len  := IROUND (Pl_VectSc * V[Y]);
                         END; {With}
                       Pl_Line ( xPix, yPix, XPix + Hz_len, yPix + Vt_len);
                     END; {For iy}
                 END; {For ix - Vector lines}

               {Draw Arrowheads}
               Pl_SetColor (PlPen_Arrow);
               FOR ix := 1 TO QuickX DO
                 BEGIN
                   xPix := HzOff + ( (ix-1)*Xblk) DIV Xlen;
                   FOR iy := 1 TO QuickY DO
                     BEGIN
                       yPix := VtOff + ( (iy-1)*Yblk) DIV Ylen;
                       WITH Dump^ [ix, iy] DO
                         BEGIN
                           Hz_len  := IROUND (Pl_VectSc * V[X]);
                           Vt_len  := IROUND (Pl_VectSc * V[Y]);
                         END; {With}
                       Pl_DrawArrow (xPix + Hz_len, yPix + Vt_len, Hz_len, Vt_len);
                     END; {For iy}
                 END; {For ix - Arrowheads}
             END; {With DumpMap^ and Scale2D}
         END; {Pl_VelVects}


     PROCEDURE PlotDump (Dmp : Word);
       VAR
         Query : Char;
       BEGIN
         Pl_InitPlot;
         Pl_SelectColour (PlPen_Axes);
         Pl_DrawWindow (Pl_LegWind);
         Pl_DrawWindow (Pl_XtrWind);
         Pl_Set2Dscale;
         CASE Sets.DumpSet OF
           Stress : BEGIN
                      Pl_StressAxes (Dmp);
                      Pl_StressLeg  (Dmp);
                    END;
           Vvect  : BEGIN
                      Pl_VelVects (Dmp);
                      Pl_VectLeg  (Dmp);
                    END;
           ESS, ISO : BEGIN
                      Pl_Contours;
                      Pl_ListLevels;
                    END;
         END; {Case}
         WITH DumpMap^ [Dmp] DO
           Pl_PlotGeom (gnum, gID);
         Pl_Descript;
         Ds_WaitMessage ('Send Plot to         Plotter (Y/N)?', Query);
         IF (Query IN ['Y', 'y'])
            THEN Pl_EndPlot(1)
            ELSE Pl_EndPlot(0);
         Pl_NamePlot;
       END; {PlotDump}

