  {Part of Dspl Unit}
   {Vector Dumps / 2-D Stress/ Velocity Vectors}

  PROCEDURE Ds_StressLeg (Dmp : Word);
    VAR
      Vstep  : Single;
      Tstr   : String;
    BEGIN
      Ds_SetWindow (LegWind);
      Vstep := LegWind.Sz[2] * 0.5;
      ClearViewPort;
      WITH DumpMap^ [Dmp] DO
        BEGIN
          SetTextStyle (SmallFont, HorizDir, FontSz);
          SetTextJustify (CenterText, CenterText);
          SetColor (LightMagenta);
          MoveTo ( LegWind.Sz[1] DIV 2, TRUNC(Vstep * (1.5/10))  );
          OutText ('PRINCIPLE STRESS');
          MoveTo ( LegWind.Sz[1] DIV 2, TRUNC(Vstep * (2.5/10))  );
          OutText ('AXES - [' + Strng (Dmp,0) + ']');
          SetTextStyle (SmallFont, HorizDir, FontSz-1);
          SetTextJustify (LeftText, CenterText);
          SetColor (DsCol_Geom);
          MoveTo ( Hinset, TRUNC(Vstep * (4/10))  );
          IF (Dim3)
             THEN OutText ('  3-D Model')
             ELSE OutText ('  2-D Model');
          MoveTo ( Hinset, TRUNC(Vstep * (5/10))  );
          OutText ('Time    = ' + Strng (Time,$0300) );
          MoveTo ( Hinset, TRUNC(Vstep * (6/10))  );
          Str (MaxVal : 12, Tstr);
          OutText ('Max Str = ' + Tstr);
          MoveTo ( Hinset, TRUNC(Vstep * (7/10))  );
          Str ( (Hpix/ScrnLen) / Ds_VectSc : 12, Tstr);
          OutText ('1CM : ' + Tstr + ' Pa');
          SetColor (DsCol_CStress);
          MoveTo ( Hinset, TRUNC(Vstep * (8/10))  );
          OutText ('--  : Compr. Stress');
          SetColor (DsCol_TStress);
          MoveTo ( Hinset, TRUNC(Vstep * (9/10))  );
          OutText ('==  : Tens.  Stress');
        END; {With}
      ResetViewPort;
    END; {Ds_StressLeg}


  PROCEDURE Ds_VectLeg (Dmp : Word);
    VAR
      Vstep  : Single;
      Tstr   : String;
    BEGIN
      Ds_SetWindow (LegWind);
      Vstep := LegWind.Sz[2] * 0.5;
      ClearViewPort;
      WITH DumpMap^ [Dmp] DO
        BEGIN
          SetTextStyle (SmallFont, HorizDir, FontSz);
          SetTextJustify (CenterText, CenterText);
          SetColor (LightMagenta);
          MoveTo ( LegWind.Sz[1] DIV 2, TRUNC(Vstep * (1.5/10))  );
          OutText ('VELOCITY VECTORS');
          MoveTo ( LegWind.Sz[1] DIV 2, TRUNC(Vstep * (2.5/10))  );
          OutText ('[' + Strng (Dmp,0) + ']');
          SetTextStyle (SmallFont, HorizDir, FontSz-1);
          SetTextJustify (LeftText, CenterText);
          SetColor (DsCol_Geom);
          MoveTo ( Hinset, TRUNC(Vstep * (4/10))  );
          IF (Dim3)
             THEN OutText ('  3-D Model')
             ELSE OutText ('  2-D Model');
          MoveTo ( Hinset, TRUNC(Vstep * (5/10))  );
          OutText ('Time    = ' + Strng (Time,$0300) );
          MoveTo ( Hinset, TRUNC(Vstep * (6/10))  );
          Str (MaxVal : 12, Tstr);
          OutText ('Max Vel = ' + Tstr);
          MoveTo ( Hinset, TRUNC(Vstep * (7/10))  );
          Str ( (Hpix/ScrnLen) / Ds_VectSc : 12, Tstr);
          OutText ('1CM : ' + Tstr + ' m/s');
        END; {With}
      ResetViewPort;
    END; {Ds_VectLeg}


  PROCEDURE Ds_StressAxes (Dmp : Word);
    VAR
      Hz_st, Hz_blk, Hz_fin, Vt_st, Vt_blk, Vt_fin : LongInt;
      Vt_sgn             : Integer;
      xPix, yPix, ix, iy : Integer;
      Hz, Vt             : LongInt;
      Maj_len,  Min_len, Angle  : Single;
    BEGIN
      SetLineType (1);
      CalcPix (Hz_st,Hz_blk,Hz_fin, Vt_st,Vt_blk,Vt_fin, Vt_sgn);
      WITH DumpMap^ [Dmp] DO
       WITH Scale2D DO
        BEGIN
          Hz_blk  := (Hz_blk*(Xqty-1)) DIV (QuickX-1);
          Vt_blk  := (Vt_blk*(Yqty-1)) DIV (QuickY-1);
          IF (Sets.VectSet)
             THEN Ds_VectSc := (Hpix/ScrnLen) * Sets.VectSc
             ELSE Ds_VectSc := Hz_blk / Xlen * Vzoom / MaxVal / VectZoom;

          {Draw Compressive Stresses}
          SetColor (DsCol_CStress);
          Hz := Hz_st;
          FOR ix := 1 TO QuickX DO
            IF (NOT(Keypressed)) THEN
               BEGIN
                 xPix := Hz DIV Xlen - 1;
                 Vt := Vt_st;
                 FOR iy := 1 TO QuickY DO
                   BEGIN
                     yPix := Vt DIV Ylen - Vt_sgn;
                     WITH Dump^ [ix, iy] DO
                       BEGIN
                         Maj_len  := IROUND (Ds_VectSc * Major);
                         Min_len  := IROUND (Ds_VectSc * Minor);
                         Angle := Ang;
                       END; {With}
                     IF (Maj_len <= 0)
                        THEN Ds_CentreBar (xPix, yPix, Maj_len, Angle);
                     IF (Min_len <= 0)
                        THEN Ds_CentreBar (xPix, yPix, Min_len, (Angle + PI*0.5));
                     Vt := Vt + Vt_blk;
                   END; {For iy}
                 Hz := Hz + Hz_blk;
               END; {For ix - Compressive Stresses}

          {Draw Tensile Stresses}
          SetColor (DsCol_TStress);
          Hz := Hz_st;
          FOR ix := 1 TO QuickX DO
            IF (NOT(Keypressed)) THEN
               BEGIN
                 xPix := Hz DIV Xlen - 1;
                 Vt := Vt_st;
                 FOR iy := 1 TO QuickY DO
                   BEGIN
                     yPix := Vt DIV Ylen - Vt_sgn;
                     WITH Dump^ [ix, iy] DO
                       BEGIN
                         Maj_len  := IROUND (Ds_VectSc * Major);
                         Min_len  := IROUND (Ds_VectSc * Minor);
                         Angle := Ang;
                       END; {With}
                     IF (Maj_len > 0)
                        THEN Ds_CentreBar (xPix, yPix, Maj_len, Angle);
                     IF (Min_len > 0)
                        THEN Ds_CentreBar (xPix, yPix, Min_len, (Angle + PI*0.5));
                     Vt := Vt + Vt_blk;
                   END; {For iy}
                 Hz := Hz + Hz_blk;
               END; {For ix - Tensile Stresses}
        END; {With DumpMap^ and Scale2D}
    END; {Ds_StressAxes}


  PROCEDURE Ds_VelVects (Dmp : Word);
    VAR
      Hz_st, Hz_blk, Hz_fin, Vt_st, Vt_blk, Vt_fin : LongInt;
      Vt_sgn             : Integer;
      xPix, yPix, ix, iy : Integer;
      Hz, Vt             : LongInt;
      Hz_len, Vt_len     : Integer;
    BEGIN
      SetLineType (1);
      CalcPix (Hz_st,Hz_blk,Hz_fin, Vt_st,Vt_blk,Vt_fin, Vt_sgn);
      WITH DumpMap^ [Dmp] DO
       WITH Scale2D DO
        BEGIN
          Hz_blk  := (Hz_blk*(Xqty-1)) DIV (QuickX-1);
          Vt_blk  := (Vt_blk*(Yqty-1)) DIV (QuickY-1);
          IF (Sets.VectSet)
             THEN Ds_VectSc := (Hpix/ScrnLen) * Sets.VectSc
             ELSE Ds_VectSc := Hz_blk / Xlen * Vzoom / MaxVal / VectZoom;

          {Draw Vector lines}
          SetColor (DsCol_Vect);
          Hz := Hz_st;
          FOR ix := 1 TO QuickX DO
            IF (NOT(Keypressed)) THEN
               BEGIN
                 xPix := Hz DIV Xlen - 1;
                 Vt := Vt_st;
                 FOR iy := 1 TO QuickY DO
                   BEGIN
                     yPix := Vt DIV Ylen - Vt_sgn;
                     WITH Dump^ [ix, iy] DO
                       BEGIN
                         Hz_len  :=   IROUND (Ds_VectSc * V[X]) DIV 2;
                         Vt_len  := - IROUND (Ds_VectSc * V[Y]) DIV 2;
                       END; {With}
                     Line (xPix-Hz_len, yPix-Vt_len, xPix+Hz_len, yPix+Vt_len);
                     Vt := Vt + Vt_blk;
                   END; {For iy}
                 Hz := Hz + Hz_blk;
               END; {For ix - Vector lines}

          {Draw Arrowheads}
          SetColor (DsCol_Arrow);
          Hz := Hz_st;
          FOR ix := 1 TO QuickX DO
            IF (NOT(Keypressed)) THEN
               BEGIN
                 xPix := Hz DIV Xlen - 1;
                 Vt := Vt_st;
                 FOR iy := 1 TO QuickY DO
                   BEGIN
                     yPix := Vt DIV Ylen - Vt_sgn;
                     WITH Dump^ [ix, iy] DO
                       BEGIN
                         Hz_len  :=   IROUND (Ds_VectSc * V[X]) DIV 2;
                         Vt_len  := - IROUND (Ds_VectSc * V[Y]) DIV 2;
                       END; {With}
                     Ds_DrawArrow (xPix+Hz_len, yPix+Vt_len, Hz_len*2, Vt_len*2);
                     Vt := Vt + Vt_blk;
                   END; {For iy}
                 Hz := Hz + Hz_blk;
               END; {For ix - Arrowheads}
        END; {With DumpMap^ and Scale2D}
    END; {Ds_VelVects}


  PROCEDURE Ds_DumpContour (Dmp : Word);
    VAR
      ix, iy : Integer;
      Mx,Mn  : Single;
      Sshear,
       Snorm : Single;
      Frict  : Single;
      RAng   : Single;
    BEGIN
      Frict := ABS (sin(PI*85/180) / cos(PI*85/180) );
      Mx := -1e20;  Mn := +1e20;
      WITH DumpMap^ [Dmp] DO
        BEGIN
          FOR ix := 1 TO QuickX DO
            IF (NOT(Keypressed)) THEN
               FOR iy := 1 TO QuickY DO
                 BEGIN
                   WITH Dump^ [ix, iy] DO
                     BEGIN
                       Rang := Ang;
                       Sshear :=  0.5 * (Major - Minor) * sin (-2*RAng);
                       Snorm  := (0.5 * (Major + Minor) + 0.5 * (Major - Minor) * cos (-2*RAng) );
                       IF (Snorm > 0)
                          THEN Snorm := 0;
                       Snapshot^ [ix,iy] := ABS(Sshear) - (Frict * ABS(Snorm) );

                       IF (Snapshot^ [ix,iy] > Mx)
                          THEN Mx := Snapshot^ [ix,iy];
                       IF (Snapshot^ [ix,iy] < Mn)
                          THEN Mn := Snapshot^ [ix,iy];
                     END;
                 END; {For iy}
          CurSnapRec.Xqty   := QuickX;  CurSnapRec.Yqty   := QuickY;
          CurSnapRec.QuickX := QuickX;  CurSnapRec.QuickY := QuickY;
          CurSnapRec.MaxVal := Mx;      CurSnapRec.MinVal := Mn;
          CurSnapRec.Time   := Time;    CurSnapRec.SnapVar := 1;
          CurSnapRec.Snp    := Dmp;
          Sets.SnapSet      := ContMap;
        END; {With}
    END; {Ds_DumpContour}

  PROCEDURE Ds_GraphDump;
    BEGIN
      Ds_SetWindow(SnapWind);
      IF NOT(Keypressed) THEN DI_ReadDump2 (CurDump);
      CASE Sets.DumpSet OF
        Stress, Vvect : Ds_ResetPalette;
        ESS, ISO      : Ds_SetPalette;
      END; {Case}
      IF NOT(Keypressed) THEN
         CASE Sets.DumpSet OF
           Stress   : BEGIN  Ds_StressAxes (CurDump); END;
           Vvect    : BEGIN  Ds_VelVects (CurDump);   END;
           ESS, ISO : BEGIN  Sets.Quick:=True; Ds_GraphSnap; END;
         END; {Case}
      CASE Sets.DumpSet OF
        Stress   : BEGIN  Ds_StressLeg  (CurDump);  END;
        Vvect    : BEGIN  Ds_VectLeg  (CurDump);    END;
        ESS, ISO : BEGIN  Ds_ListLevels;            END;
      END; {Case}
      ResetViewPort;
      IF (Sets.ShowTitles) THEN Ds_Titles;
    END; {Ds_GraphDump}

